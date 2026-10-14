// 128/256-bit LSX/LASX tree force and neighbor search kernels for the
// LoongArch architecture (Loongson-3A5000/3A6000, LA464/LA664 cores).
// The kernels are API-compatible with the reference (NoSimd) kernels in
// soft_force.hpp and are enabled by ./configure --with-arch=loongarch
// (--with-simd=lsx or lasx) (-D USE_LARCH). The kernels use float (not
// PS::F32) for the SIMD intrinsics, since PS::F32 can be double under
// PARTICLE_SIMULATOR_ALL_64BIT_PRECISION.
// The vector width is 4 lanes for LSX (-D LARCH_SIMD_LSX) and 8 lanes for
// LASX (default). 1/sqrt() uses the exact hardware vector instruction
// (vfrsqrt.s / xvfrsqrt.s) by default; -D LARCH_RSQRT_SW selects a
// software chain (bit-magic seed + Newton steps) instead.

#pragma once
#ifdef USE_LARCH

#include <lsxintrin.h>
#include <lasxintrin.h>
#include <vector>
#include"soft_ptcl.hpp"
#include"soft_force.hpp"

#ifdef USE_QUAD
#ifdef FDPS_V8
#define SPJSoft PS::SPJQuadrupole
#else
#define SPJSoft PS::SPJQuadrupoleInAndOut
#endif
#else
#ifdef FDPS_V8
#define SPJSoft PS::SPJMonopole
#else
#define SPJSoft PS::SPJMonopoleInAndOut
#endif
#endif

//! vector width (lanes): 4 for LSX, 8 for LASX
#ifdef LARCH_SIMD_LSX
static const PS::S32 VW = 4;
#else
static const PS::S32 VW = 8;
#endif

//! position used to pad inactive lanes; (1e15)^2 stays finite in float
static const float PAD_POS = 1e15f;

#ifdef LARCH_SIMD_LSX
using VF = __m128;    //! 4 x float
using VI = __m128i;   //! 4 x int32
#define LARCH_VLD(p)          ((VF)__lsx_vld((p), 0))
#define LARCH_VST(v, p)       __lsx_vst((__m128i)(v), (p), 0)
#define LARCH_VADD(a, b)      __lsx_vfadd_s((a), (b))
#define LARCH_VSUB(a, b)      __lsx_vfsub_s((a), (b))
#define LARCH_VMUL(a, b)      __lsx_vfmul_s((a), (b))
#define LARCH_VMADD(a, b, c)  __lsx_vfmadd_s((a), (b), (c))
#define LARCH_VNMSUB(a, b, c) __lsx_vfnmsub_s((a), (b), (c))
#define LARCH_VMAX(a, b)      __lsx_vfmax_s((a), (b))
#define LARCH_VCMPLT(a, b)    __lsx_vfcmp_clt_s((a), (b))
#define LARCH_VAND(a, b)      __lsx_vand_v((a), (b))
#define LARCH_VOR(a, b)       __lsx_vor_v((a), (b))
#define LARCH_VADDW(a, b)     __lsx_vadd_w((a), (b))
#define LARCH_VSUBW(a, b)     __lsx_vsub_w((a), (b))
#define LARCH_VSRLIW(a, n)    __lsx_vsrli_w((a), (n))
static inline VF vrepl(const float x){ return (VF)__lsx_vldrepl_w(&x, 0); }
static inline VF vrepl_mem(const float* p){ return (VF)__lsx_vldrepl_w(p, 0); }
static inline VI vrepli(const PS::S32 x){ return (VI)__lsx_vldrepl_w(&x, 0); }
#else
using VF = __m256;    //! 8 x float
using VI = __m256i;   //! 8 x int32
#define LARCH_VLD(p)          ((VF)__lasx_xvld((p), 0))
#define LARCH_VST(v, p)       __lasx_xvst((__m256i)(v), (p), 0)
#define LARCH_VADD(a, b)      __lasx_xvfadd_s((a), (b))
#define LARCH_VSUB(a, b)      __lasx_xvfsub_s((a), (b))
#define LARCH_VMUL(a, b)      __lasx_xvfmul_s((a), (b))
#define LARCH_VMADD(a, b, c)  __lasx_xvfmadd_s((a), (b), (c))
#define LARCH_VNMSUB(a, b, c) __lasx_xvfnmsub_s((a), (b), (c))
#define LARCH_VMAX(a, b)      __lasx_xvfmax_s((a), (b))
#define LARCH_VCMPLT(a, b)    __lasx_xvfcmp_clt_s((a), (b))
#define LARCH_VAND(a, b)      __lasx_xvand_v((a), (b))
#define LARCH_VOR(a, b)       __lasx_xvor_v((a), (b))
#define LARCH_VADDW(a, b)     __lasx_xvadd_w((a), (b))
#define LARCH_VSUBW(a, b)     __lasx_xvsub_w((a), (b))
#define LARCH_VSRLIW(a, n)    __lasx_xvsrli_w((a), (n))
static inline VF vrepl(const float x){ return (VF)__lasx_xvldrepl_w(&x, 0); }
static inline VF vrepl_mem(const float* p){ return (VF)__lasx_xvldrepl_w(p, 0); }
static inline VI vrepli(const PS::S32 x){ return (VI)__lasx_xvldrepl_w(&x, 0); }
#endif

static inline VI vzeroi(){ return vrepli(0); }
static inline VF vzerof(){ return vrepl(0.0f); }

//! horizontal sum of a float vector
static inline float hadd(const VF v){
    float t[VW];
    LARCH_VST(v, t);
    float s = 0.0f;
    for(PS::S32 k=0;k<VW;k++) s += t[k];
    return s;
}
//! horizontal sum of an int vector
static inline PS::S32 haddi(const VI v){
    PS::S32 t[VW];
    LARCH_VST((VF)v, t);
    PS::S32 s = 0;
    for(PS::S32 k=0;k<VW;k++) s += t[k];
    return s;
}

//! fast reciprocal sqrt (VW x float).
//! Default: the exact hardware vector instruction (vfrsqrt.s / xvfrsqrt.s);
//! on the LA664 it returns the correctly rounded float result, so the
//! kernels add no rsqrt error on top of the float force arithmetic.
//! LARCH_RSQRT_SW: bit-magic seed + x86 phantomquad Newton step + FUGAKU
//! cubic correction (max relative error ~1.5e-7 on the 3A6000); useful for
//! experiments, kept as a build option.
static inline VF rsqrt_sw(const VF a){
    const PS::S32 magic = 0x5f3759df;
    VI ib = (VI)a;
    ib = LARCH_VSUBW(vrepli(magic), LARCH_VSRLIW(ib, 1));
    VF x = (VF)ib;
    const VF half = vrepl(0.5f), one = vrepl(1.0f), three = vrepl(3.0f);
    // x86 RSQRT_NR_EPJ_X2: x = x*(3 - a*x*x)*0.5
    VF x2 = LARCH_VMUL(x, x);
    VF h = LARCH_VNMSUB(a, x2, three);
    x = LARCH_VMUL(LARCH_VMUL(x, h), half);
    // FUGAKU cubic: h = 1 - a*x*x; x = x + x*((0.375*h + 0.5)*h)
    x2 = LARCH_VMUL(x, x);
    h = LARCH_VNMSUB(a, x2, one);
    VF poly = LARCH_VMADD(h, vrepl(0.375f), half);
    poly = LARCH_VMUL(poly, h);
    return LARCH_VMADD(x, poly, x);
}

#if defined(LARCH_RSQRT_SW) && !defined(LARCH_RSQRT_HW)
static inline VF rsqrtN(const VF a){ return rsqrt_sw(a); }
#else
static inline VF rsqrtN(const VF a){
#ifdef LARCH_SIMD_LSX
    return (VF)__lsx_vfrsqrt_s(a);
#else
    return (VF)__lasx_xvfrsqrt_s(a);
#endif
}
#endif

//! per-thread scratch buffers (structure of arrays), reused across calls
struct Scratch{
    std::vector<PS::S32> lst;                           //! original indices of active i
    std::vector<float>   ix, iy, iz, irs;               //! compact i (padded to a multiple of VW)
    std::vector<float>   jx, jy, jz, jm, jrs;           //! compact j for I1_JV
    std::vector<float>   qxx, qyy, qzz, qxy, qxz, qyz;  //! compact super-particle quadrupole
};
//! return the thread-local scratch buffer
static inline Scratch& scratch(){
    static thread_local Scratch s;
    return s;
}

//! compact the active EPI list (type==1) into SoA floats, padded to a
//! multiple of VW; when shift is true the positions are translated by
//! (cx,cy,cz) to suppress float cancellation (same filter as the x86
//! SIMD/Fugaku/Tsv110 kernels)
static inline PS::S32 compact_i(const EPISoft* epi, const PS::S32 ni, Scratch& S,
                                const bool shift,
                                const PS::F64 cx, const PS::F64 cy, const PS::F64 cz){
    S.lst.clear(); S.ix.clear(); S.iy.clear(); S.iz.clear(); S.irs.clear();
    for(PS::S32 i=0;i<ni;i++){
        if(epi[i].type!=1) continue;
        S.lst.push_back(i);
        S.ix.push_back((float)(epi[i].pos.x - (shift ? cx : 0.0)));
        S.iy.push_back((float)(epi[i].pos.y - (shift ? cy : 0.0)));
        S.iz.push_back((float)(epi[i].pos.z - (shift ? cz : 0.0)));
        const float rs = (float)epi[i].r_search;
        S.irs.push_back(rs);
    }
    const PS::S32 na = (PS::S32)S.lst.size();
    const PS::S32 npad = (na+VW-1)/VW*VW;
    for(PS::S32 k=na;k<npad;k++){
        S.ix.push_back(PAD_POS); S.iy.push_back(PAD_POS); S.iz.push_back(PAD_POS); S.irs.push_back(0.f);
    }
    return npad;
}

//! compact the EPJ list for the EP-EP force into SoA floats, translated by
//! the origin (cx,cy,cz) and skipping zero-mass entries (same behaviour as
//! the x86 SIMD/Fugaku/Tsv110 kernels); padded to a multiple of VW + VW
static inline PS::S32 compact_j_ep(const EPJSoft* epj, const PS::S32 nj, Scratch& S,
                                   const PS::F64 cx, const PS::F64 cy, const PS::F64 cz){
    S.jx.clear(); S.jy.clear(); S.jz.clear(); S.jm.clear(); S.jrs.clear();
    for(PS::S32 j=0;j<nj;j++){
        if(epj[j].mass<=0.0) continue;
        S.jx.push_back((float)(epj[j].pos.x-cx)); S.jy.push_back((float)(epj[j].pos.y-cy)); S.jz.push_back((float)(epj[j].pos.z-cz));
        S.jm.push_back((float)epj[j].mass);
        S.jrs.push_back((float)(epj[j].r_search*epj[j].r_search));
    }
    const PS::S32 na = (PS::S32)S.jx.size();
    const PS::S32 npad = (na+VW-1)/VW*VW + VW;
    for(PS::S32 k=na;k<npad;k++){
        S.jx.push_back(PAD_POS); S.jy.push_back(PAD_POS); S.jz.push_back(PAD_POS); S.jm.push_back(0.f); S.jrs.push_back(0.f);
    }
    return npad;
}

//! compact the EPJ list for the neighbor search into SoA floats (no mass
//! filter and no shift, matching the reference kernel semantics); padded to
//! a multiple of VW + VW
static inline PS::S32 compact_j_nb(const EPJSoft* epj, const PS::S32 nj, Scratch& S){
    S.jx.clear(); S.jy.clear(); S.jz.clear(); S.jm.clear(); S.jrs.clear();
    for(PS::S32 j=0;j<nj;j++){
        S.jx.push_back((float)epj[j].pos.x); S.jy.push_back((float)epj[j].pos.y); S.jz.push_back((float)epj[j].pos.z);
        S.jm.push_back((float)epj[j].mass);
        S.jrs.push_back((float)(epj[j].r_search*epj[j].r_search));
    }
    const PS::S32 na = (PS::S32)S.jx.size();
    const PS::S32 npad = (na+VW-1)/VW*VW + VW;
    for(PS::S32 k=na;k<npad;k++){
        S.jx.push_back(PAD_POS); S.jy.push_back(PAD_POS); S.jz.push_back(PAD_POS); S.jm.push_back(0.f); S.jrs.push_back(0.f);
    }
    return npad;
}

//! compact the SPJ list for the EP-SP force into SoA floats (translated by
//! the origin cx,cy,cz), padded to a multiple of VW + VW
template <class Tsp>
static inline PS::S32 compact_j_sp(const Tsp* spj, const PS::S32 nj, Scratch& S,
                                   const PS::F64 cx, const PS::F64 cy, const PS::F64 cz,
                                   const bool quad){
    S.jx.clear(); S.jy.clear(); S.jz.clear(); S.jm.clear();
    if(quad){ S.qxx.clear(); S.qyy.clear(); S.qzz.clear(); S.qxy.clear(); S.qxz.clear(); S.qyz.clear(); }
    for(PS::S32 j=0;j<nj;j++){
        S.jx.push_back((float)(spj[j].pos.x-cx)); S.jy.push_back((float)(spj[j].pos.y-cy)); S.jz.push_back((float)(spj[j].pos.z-cz));
        S.jm.push_back((float)spj[j].mass);
        if(quad){
            S.qxx.push_back((float)spj[j].quad.xx); S.qyy.push_back((float)spj[j].quad.yy); S.qzz.push_back((float)spj[j].quad.zz);
            S.qxy.push_back((float)spj[j].quad.xy); S.qxz.push_back((float)spj[j].quad.xz); S.qyz.push_back((float)spj[j].quad.yz);
        }
    }
    const PS::S32 npad = (nj+VW-1)/VW*VW + VW;
    for(PS::S32 k=nj;k<npad;k++){
        S.jx.push_back(PAD_POS); S.jy.push_back(PAD_POS); S.jz.push_back(PAD_POS); S.jm.push_back(0.f);
        if(quad){
            S.qxx.push_back(0.f); S.qyy.push_back(0.f); S.qzz.push_back(0.f);
            S.qxy.push_back(0.f); S.qxz.push_back(0.f); S.qyz.push_back(0.f);
        }
    }
    return npad;
}

//! kernel orientation selector: vectorize over i (IV_J1) or over j (I1_JV)
static inline bool use_iv(const PS::S32 ni, const PS::S32 nj){
    return (nj<=2*VW) || (ni<=VW) || ((PS::S64)ni*(PS::S64)nj<=64*VW);
}
//! below this interaction count the scalar reference kernel is faster
static inline bool use_scalar(const PS::S32 ni, const PS::S32 nj){
    return ((PS::S64)ni*(PS::S64)nj < 32*VW);
}

//! tree neighbor search (EP-EP); counting semantics follow the reference
//! kernel (float r2 < max(r_search_i, r_search_j)^2), no origin shift
struct SearchNeighborEpEpLarch{
    void operator()(const EPISoft* epi, const PS::S32 ni, const EPJSoft* epj, const PS::S32 nj, ForceSoft* force) const {
        if(use_scalar(ni,nj)){ SearchNeighborEpEpNoSimd()(epi,ni,epj,nj,force); return; }
        if(use_iv(ni,nj)) Kernel_IV_J1(epi,ni,epj,nj,force);
        else              Kernel_I1_JV(epi,ni,epj,nj,force);
    }
    void Kernel_IV_J1(const EPISoft* epi, const PS::S32 ni, const EPJSoft* epj, const PS::S32 nj, ForceSoft* force) const {
        Scratch& S = scratch();
        const PS::S32 npad = compact_i(epi,ni,S,false,0,0,0);
        const PS::S32 na = (PS::S32)S.lst.size();
        S.jx.resize(nj); S.jy.resize(nj); S.jz.resize(nj); S.jrs.resize(nj);
        for(PS::S32 j=0;j<nj;j++){
            S.jx[j]=(float)epj[j].pos.x; S.jy[j]=(float)epj[j].pos.y; S.jz[j]=(float)epj[j].pos.z;
            S.jrs[j]=(float)(epj[j].r_search*epj[j].r_search);
        }
        for(PS::S32 ib=0; ib<npad; ib+=VW){
            const VF xi=LARCH_VLD(&S.ix[ib]), yi=LARCH_VLD(&S.iy[ib]), zi=LARCH_VLD(&S.iz[ib]);
            const VF rsi=LARCH_VLD(&S.irs[ib]);
            const VF rsi2=LARCH_VMUL(rsi,rsi);
            VI nn=vzeroi();
            for(PS::S32 j=0;j<nj;j++){
                const VF dx=LARCH_VSUB(xi,vrepl_mem(&S.jx[j]));
                const VF dy=LARCH_VSUB(yi,vrepl_mem(&S.jy[j]));
                const VF dz=LARCH_VSUB(zi,vrepl_mem(&S.jz[j]));
                VF r2=LARCH_VMADD(dx,dx,vzerof());
                r2=LARCH_VMADD(dy,dy,r2);
                r2=LARCH_VMADD(dz,dz,r2);
                const VF R2=LARCH_VMAX(rsi2,vrepl_mem(&S.jrs[j]));
                nn=LARCH_VADDW(nn,LARCH_VAND(LARCH_VCMPLT(r2,R2),vrepli(1)));
            }
            PS::S32 tmp[VW];
            LARCH_VST((VF)nn,tmp);
            for(PS::S32 k=0;k<VW;k++){
                const PS::S32 ii=ib+k; if(ii>=na) break;
                force[S.lst[ii]].n_ngb = tmp[k];
            }
        }
    }
    void Kernel_I1_JV(const EPISoft* epi, const PS::S32 ni, const EPJSoft* epj, const PS::S32 nj, ForceSoft* force) const {
        Scratch& S = scratch();
        compact_i(epi,ni,S,false,0,0,0);
        const PS::S32 na = (PS::S32)S.lst.size();
        const PS::S32 npj = compact_j_nb(epj,nj,S);
        for(PS::S32 ii=0; ii<na; ii++){
            const VF xi=vrepl(S.ix[ii]), yi=vrepl(S.iy[ii]), zi=vrepl(S.iz[ii]);
            const VF rsi=vrepl(S.irs[ii]);
            const VF rsi2=LARCH_VMUL(rsi,rsi);
            VI nn=vzeroi();
            for(PS::S32 j=0;j<npj;j+=VW){
                const VF dx=LARCH_VSUB(xi,LARCH_VLD(&S.jx[j]));
                const VF dy=LARCH_VSUB(yi,LARCH_VLD(&S.jy[j]));
                const VF dz=LARCH_VSUB(zi,LARCH_VLD(&S.jz[j]));
                VF r2=LARCH_VMADD(dx,dx,vzerof());
                r2=LARCH_VMADD(dy,dy,r2);
                r2=LARCH_VMADD(dz,dz,r2);
                const VF R2=LARCH_VMAX(rsi2,LARCH_VLD(&S.jrs[j]));
                nn=LARCH_VADDW(nn,LARCH_VAND(LARCH_VCMPLT(r2,R2),vrepli(1)));
            }
            force[S.lst[ii]].n_ngb = haddi(nn);
        }
    }
};

//! EP-EP force with linear cutoff and neighbor counting; positions are
//! translated by the first active i-particle to suppress float cancellation
struct CalcForceEpEpWithLinearCutoffLarch{
    float eps2, rcut2; PS::F64 G;
    CalcForceEpEpWithLinearCutoffLarch(){}
    CalcForceEpEpWithLinearCutoffLarch(PS::F64 e2, PS::F64 rc2, PS::F64 g):eps2((float)e2),rcut2((float)rc2),G(g){}
    void operator()(const EPISoft* epi, const PS::S32 ni, const EPJSoft* epj, const PS::S32 nj, ForceSoft* force) const {
        if(use_scalar(ni,nj)){ CalcForceEpEpWithLinearCutoffNoSimd()(epi,ni,epj,nj,force); return; }
        if(use_iv(ni,nj)) Kernel_IV_J1(epi,ni,epj,nj,force);
        else              Kernel_I1_JV(epi,ni,epj,nj,force);
    }
    void Kernel_IV_J1(const EPISoft* epi, const PS::S32 ni, const EPJSoft* epj, const PS::S32 nj, ForceSoft* force) const {
        Scratch& S = scratch();
        PS::F64 cx=0, cy=0, cz=0;
        for(PS::S32 i=0;i<ni;i++){
            if(epi[i].type==1){ cx=epi[i].pos.x; cy=epi[i].pos.y; cz=epi[i].pos.z; break; }
        }
        const PS::S32 npad = compact_i(epi,ni,S,true,cx,cy,cz);
        const PS::S32 na = (PS::S32)S.lst.size();
        const PS::S32 nja = compact_j_ep(epj,nj,S,cx,cy,cz);
        const VF e2v=vrepl(eps2), rcv=vrepl(rcut2);
        for(PS::S32 ib=0; ib<npad; ib+=VW){
            const VF xi=LARCH_VLD(&S.ix[ib]), yi=LARCH_VLD(&S.iy[ib]), zi=LARCH_VLD(&S.iz[ib]);
            const VF rsi=LARCH_VLD(&S.irs[ib]);
            const VF rsi2=LARCH_VMUL(rsi,rsi);
            VF ax=vzerof(), ay=vzerof(), az=vzerof(), ap=vzerof();
            VI nn=vzeroi();
            for(PS::S32 j=0;j<nja;j++){
                const VF dx=LARCH_VSUB(xi,vrepl_mem(&S.jx[j]));
                const VF dy=LARCH_VSUB(yi,vrepl_mem(&S.jy[j]));
                const VF dz=LARCH_VSUB(zi,vrepl_mem(&S.jz[j]));
                const VF mj=vrepl_mem(&S.jm[j]);
                VF r2=LARCH_VMADD(dx,dx,vzerof());
                r2=LARCH_VMADD(dy,dy,r2);
                r2=LARCH_VMADD(dz,dz,r2);
                const VF R2=LARCH_VMAX(rsi2,vrepl_mem(&S.jrs[j]));
                nn=LARCH_VADDW(nn,LARCH_VAND(LARCH_VCMPLT(r2,R2),vrepli(1)));
                const VF rc=LARCH_VMAX(LARCH_VADD(r2,e2v),rcv);
                const VF ri=rsqrtN(rc);
                const VF ri2=LARCH_VMUL(ri,ri);
                const VF mr=LARCH_VMUL(mj,ri);
                const VF mr3=LARCH_VMUL(ri2,mr);
                ax=LARCH_VNMSUB(mr3,dx,ax);
                ay=LARCH_VNMSUB(mr3,dy,ay);
                az=LARCH_VNMSUB(mr3,dz,az);
                ap=LARCH_VSUB(ap,mr);
            }
            float fx[VW],fy[VW],fz[VW],fp[VW]; PS::S32 fn[VW];
            LARCH_VST(ax,fx); LARCH_VST(ay,fy); LARCH_VST(az,fz); LARCH_VST(ap,fp); LARCH_VST((VF)nn,fn);
            for(PS::S32 k=0;k<VW;k++){
                const PS::S32 ii=ib+k; if(ii>=na) break;
                const PS::S32 jj=S.lst[ii];
                force[jj].acc.x += G*fx[k];
                force[jj].acc.y += G*fy[k];
                force[jj].acc.z += G*fz[k];
                force[jj].pot   += G*fp[k];
                force[jj].n_ngb  = fn[k];
            }
        }
    }
    void Kernel_I1_JV(const EPISoft* epi, const PS::S32 ni, const EPJSoft* epj, const PS::S32 nj, ForceSoft* force) const {
        Scratch& S = scratch();
        PS::F64 cx=0, cy=0, cz=0;
        for(PS::S32 i=0;i<ni;i++){
            if(epi[i].type==1){ cx=epi[i].pos.x; cy=epi[i].pos.y; cz=epi[i].pos.z; break; }
        }
        compact_i(epi,ni,S,true,cx,cy,cz);
        const PS::S32 na = (PS::S32)S.lst.size();
        const PS::S32 npj = compact_j_ep(epj,nj,S,cx,cy,cz);
        const VF e2v=vrepl(eps2), rcv=vrepl(rcut2);
        for(PS::S32 ii=0; ii<na; ii++){
            const VF xi=vrepl(S.ix[ii]), yi=vrepl(S.iy[ii]), zi=vrepl(S.iz[ii]);
            const VF rsi=vrepl(S.irs[ii]);
            const VF rsi2=LARCH_VMUL(rsi,rsi);
            VF ax=vzerof(), ay=vzerof(), az=vzerof(), ap=vzerof();
            VI nn=vzeroi();
            for(PS::S32 j=0;j<npj;j+=VW){
                const VF dx=LARCH_VSUB(xi,LARCH_VLD(&S.jx[j]));
                const VF dy=LARCH_VSUB(yi,LARCH_VLD(&S.jy[j]));
                const VF dz=LARCH_VSUB(zi,LARCH_VLD(&S.jz[j]));
                const VF mj=LARCH_VLD(&S.jm[j]);
                VF r2=LARCH_VMADD(dx,dx,vzerof());
                r2=LARCH_VMADD(dy,dy,r2);
                r2=LARCH_VMADD(dz,dz,r2);
                const VF R2=LARCH_VMAX(rsi2,LARCH_VLD(&S.jrs[j]));
                nn=LARCH_VADDW(nn,LARCH_VAND(LARCH_VCMPLT(r2,R2),vrepli(1)));
                const VF rc=LARCH_VMAX(LARCH_VADD(r2,e2v),rcv);
                const VF ri=rsqrtN(rc);
                const VF ri2=LARCH_VMUL(ri,ri);
                const VF mr=LARCH_VMUL(mj,ri);
                const VF mr3=LARCH_VMUL(ri2,mr);
                ax=LARCH_VNMSUB(mr3,dx,ax);
                ay=LARCH_VNMSUB(mr3,dy,ay);
                az=LARCH_VNMSUB(mr3,dz,az);
                ap=LARCH_VSUB(ap,mr);
            }
            const PS::S32 jj=S.lst[ii];
            force[jj].acc.x += G*hadd(ax);
            force[jj].acc.y += G*hadd(ay);
            force[jj].acc.z += G*hadd(az);
            force[jj].pot   += G*hadd(ap);
            force[jj].n_ngb  = haddi(nn);
        }
    }
};

//! EP-SP monopole force (origin shift by the first active i-particle)
struct CalcForceEpSpMonoLarch{
    float eps2; PS::F64 G;
    CalcForceEpSpMonoLarch(){}
    CalcForceEpSpMonoLarch(PS::F64 e2, PS::F64 g):eps2((float)e2),G(g){}
    template <class Tsp>
    void operator()(const EPISoft* epi, const PS::S32 ni, const Tsp* spj, const PS::S32 nj, ForceSoft* force) const {
        if(use_scalar(ni,nj)){ CalcForceEpSpMonoNoSimd()(epi,ni,spj,nj,force); return; }
        if(use_iv(ni,nj)) Kernel_IV_J1(epi,ni,spj,nj,force);
        else              Kernel_I1_JV(epi,ni,spj,nj,force);
    }
    template <class Tsp>
    void Kernel_IV_J1(const EPISoft* epi, const PS::S32 ni, const Tsp* spj, const PS::S32 nj, ForceSoft* force) const {
        Scratch& S = scratch();
        PS::F64 cx=0, cy=0, cz=0;
        for(PS::S32 i=0;i<ni;i++){
            if(epi[i].type==1){ cx=epi[i].pos.x; cy=epi[i].pos.y; cz=epi[i].pos.z; break; }
        }
        const PS::S32 npad = compact_i(epi,ni,S,true,cx,cy,cz);
        const PS::S32 na = (PS::S32)S.lst.size();
        compact_j_sp(spj,nj,S,cx,cy,cz,false);
        const VF e2v=vrepl(eps2);
        for(PS::S32 ib=0; ib<npad; ib+=VW){
            const VF xi=LARCH_VLD(&S.ix[ib]), yi=LARCH_VLD(&S.iy[ib]), zi=LARCH_VLD(&S.iz[ib]);
            VF ax=vzerof(), ay=vzerof(), az=vzerof(), ap=vzerof();
            for(PS::S32 j=0;j<nj;j++){
                const VF dx=LARCH_VSUB(xi,vrepl_mem(&S.jx[j]));
                const VF dy=LARCH_VSUB(yi,vrepl_mem(&S.jy[j]));
                const VF dz=LARCH_VSUB(zi,vrepl_mem(&S.jz[j]));
                const VF mj=vrepl_mem(&S.jm[j]);
                VF r2=LARCH_VMADD(dx,dx,e2v);
                r2=LARCH_VMADD(dy,dy,r2);
                r2=LARCH_VMADD(dz,dz,r2);
                const VF ri=rsqrtN(r2);
                const VF ri2=LARCH_VMUL(ri,ri);
                const VF mr=LARCH_VMUL(mj,ri);
                const VF mr3=LARCH_VMUL(ri2,mr);
                ax=LARCH_VNMSUB(mr3,dx,ax);
                ay=LARCH_VNMSUB(mr3,dy,ay);
                az=LARCH_VNMSUB(mr3,dz,az);
                ap=LARCH_VSUB(ap,mr);
            }
            float fx[VW],fy[VW],fz[VW],fp[VW];
            LARCH_VST(ax,fx); LARCH_VST(ay,fy); LARCH_VST(az,fz); LARCH_VST(ap,fp);
            for(PS::S32 k=0;k<VW;k++){
                const PS::S32 ii=ib+k; if(ii>=na) break;
                const PS::S32 jj=S.lst[ii];
                force[jj].acc.x += G*fx[k];
                force[jj].acc.y += G*fy[k];
                force[jj].acc.z += G*fz[k];
                force[jj].pot   += G*fp[k];
            }
        }
    }
    template <class Tsp>
    void Kernel_I1_JV(const EPISoft* epi, const PS::S32 ni, const Tsp* spj, const PS::S32 nj, ForceSoft* force) const {
        Scratch& S = scratch();
        PS::F64 cx=0, cy=0, cz=0;
        for(PS::S32 i=0;i<ni;i++){
            if(epi[i].type==1){ cx=epi[i].pos.x; cy=epi[i].pos.y; cz=epi[i].pos.z; break; }
        }
        compact_i(epi,ni,S,true,cx,cy,cz);
        const PS::S32 na = (PS::S32)S.lst.size();
        const PS::S32 npj = compact_j_sp(spj,nj,S,cx,cy,cz,false);
        const VF e2v=vrepl(eps2);
        for(PS::S32 ii=0; ii<na; ii++){
            const VF xi=vrepl(S.ix[ii]), yi=vrepl(S.iy[ii]), zi=vrepl(S.iz[ii]);
            VF ax=vzerof(), ay=vzerof(), az=vzerof(), ap=vzerof();
            for(PS::S32 j=0;j<npj;j+=VW){
                const VF dx=LARCH_VSUB(xi,LARCH_VLD(&S.jx[j]));
                const VF dy=LARCH_VSUB(yi,LARCH_VLD(&S.jy[j]));
                const VF dz=LARCH_VSUB(zi,LARCH_VLD(&S.jz[j]));
                const VF mj=LARCH_VLD(&S.jm[j]);
                VF r2=LARCH_VMADD(dx,dx,e2v);
                r2=LARCH_VMADD(dy,dy,r2);
                r2=LARCH_VMADD(dz,dz,r2);
                const VF ri=rsqrtN(r2);
                const VF ri2=LARCH_VMUL(ri,ri);
                const VF mr=LARCH_VMUL(mj,ri);
                const VF mr3=LARCH_VMUL(ri2,mr);
                ax=LARCH_VNMSUB(mr3,dx,ax);
                ay=LARCH_VNMSUB(mr3,dy,ay);
                az=LARCH_VNMSUB(mr3,dz,az);
                ap=LARCH_VSUB(ap,mr);
            }
            const PS::S32 jj=S.lst[ii];
            force[jj].acc.x += G*hadd(ax);
            force[jj].acc.y += G*hadd(ay);
            force[jj].acc.z += G*hadd(az);
            force[jj].pot   += G*hadd(ap);
        }
    }
};

//! EP-SP quadrupole force (origin shift by the first active i-particle)
struct CalcForceEpSpQuadLarch{
    float eps2; PS::F64 G;
    CalcForceEpSpQuadLarch(){}
    CalcForceEpSpQuadLarch(PS::F64 e2, PS::F64 g):eps2((float)e2),G(g){}
    template <class Tsp>
    void operator()(const EPISoft* epi, const PS::S32 ni, const Tsp* spj, const PS::S32 nj, ForceSoft* force) const {
        if(use_scalar(ni,nj)){ CalcForceEpSpQuadNoSimd()(epi,ni,spj,nj,force); return; }
        if(use_iv(ni,nj)) Kernel_IV_J1(epi,ni,spj,nj,force);
        else              Kernel_I1_JV(epi,ni,spj,nj,force);
    }
    template <class Tsp>
    void Kernel_IV_J1(const EPISoft* epi, const PS::S32 ni, const Tsp* spj, const PS::S32 nj, ForceSoft* force) const {
        Scratch& S = scratch();
        PS::F64 cx=0, cy=0, cz=0;
        for(PS::S32 i=0;i<ni;i++){
            if(epi[i].type==1){ cx=epi[i].pos.x; cy=epi[i].pos.y; cz=epi[i].pos.z; break; }
        }
        const PS::S32 npad = compact_i(epi,ni,S,true,cx,cy,cz);
        const PS::S32 na = (PS::S32)S.lst.size();
        compact_j_sp(spj,nj,S,cx,cy,cz,true);
        const VF e2v=vrepl(eps2), half=vrepl(0.5f), five=vrepl(5.0f), v15=vrepl(1.5f), m2=vrepl(-2.0f);
        for(PS::S32 ib=0; ib<npad; ib+=VW){
            const VF xi=LARCH_VLD(&S.ix[ib]), yi=LARCH_VLD(&S.iy[ib]), zi=LARCH_VLD(&S.iz[ib]);
            VF ax=vzerof(), ay=vzerof(), az=vzerof(), ap=vzerof();
            for(PS::S32 j=0;j<nj;j++){
                const VF dx=LARCH_VSUB(xi,vrepl_mem(&S.jx[j]));
                const VF dy=LARCH_VSUB(yi,vrepl_mem(&S.jy[j]));
                const VF dz=LARCH_VSUB(zi,vrepl_mem(&S.jz[j]));
                const VF mj=vrepl_mem(&S.jm[j]);
                const VF qxx=vrepl_mem(&S.qxx[j]), qyy=vrepl_mem(&S.qyy[j]), qzz=vrepl_mem(&S.qzz[j]);
                const VF qxy=vrepl_mem(&S.qxy[j]), qxz=vrepl_mem(&S.qxz[j]), qyz=vrepl_mem(&S.qyz[j]);
                VF r2=LARCH_VMADD(dx,dx,e2v);
                r2=LARCH_VMADD(dy,dy,r2);
                r2=LARCH_VMADD(dz,dz,r2);
                const VF ri=rsqrtN(r2);
                const VF ri2=LARCH_VMUL(ri,ri);
                const VF ri3=LARCH_VMUL(ri2,ri);
                const VF r5=LARCH_VMUL(LARCH_VMUL(ri2,ri3),v15);
                const VF tr=LARCH_VADD(LARCH_VADD(qxx,qyy),qzz);
                const VF qrx=LARCH_VMADD(qxz,dz,LARCH_VMADD(qxy,dy,LARCH_VMUL(qxx,dx)));
                const VF qry=LARCH_VMADD(qxy,dx,LARCH_VMADD(qyz,dz,LARCH_VMUL(qyy,dy)));
                const VF qrz=LARCH_VMADD(qyz,dy,LARCH_VMADD(qxz,dx,LARCH_VMUL(qzz,dz)));
                const VF qrr=LARCH_VMADD(qrz,dz,LARCH_VMADD(qry,dy,LARCH_VMUL(qrx,dx)));
                const VF qrr5=LARCH_VMUL(r5,qrr);
                const VF qrr7=LARCH_VMUL(ri2,qrr5);
                const VF A=LARCH_VADD(LARCH_VSUB(LARCH_VMUL(mj,ri3),LARCH_VMUL(tr,r5)),LARCH_VMUL(five,qrr7));
                const VF m2r5=LARCH_VMUL(m2,r5);
                ax=LARCH_VNMSUB(A,dx,ax); ay=LARCH_VNMSUB(A,dy,ay); az=LARCH_VNMSUB(A,dz,az);
                ax=LARCH_VNMSUB(m2r5,qrx,ax); ay=LARCH_VNMSUB(m2r5,qry,ay); az=LARCH_VNMSUB(m2r5,qrz,az);
                ap=LARCH_VSUB(ap,LARCH_VMUL(mj,ri));
                ap=LARCH_VADD(ap,LARCH_VMUL(half,LARCH_VMUL(tr,ri3)));
                ap=LARCH_VSUB(ap,qrr5);
            }
            float fx[VW],fy[VW],fz[VW],fp[VW];
            LARCH_VST(ax,fx); LARCH_VST(ay,fy); LARCH_VST(az,fz); LARCH_VST(ap,fp);
            for(PS::S32 k=0;k<VW;k++){
                const PS::S32 ii=ib+k; if(ii>=na) break;
                const PS::S32 jj=S.lst[ii];
                force[jj].acc.x += G*fx[k];
                force[jj].acc.y += G*fy[k];
                force[jj].acc.z += G*fz[k];
                force[jj].pot   += G*fp[k];
            }
        }
    }
    template <class Tsp>
    void Kernel_I1_JV(const EPISoft* epi, const PS::S32 ni, const Tsp* spj, const PS::S32 nj, ForceSoft* force) const {
        Scratch& S = scratch();
        PS::F64 cx=0, cy=0, cz=0;
        for(PS::S32 i=0;i<ni;i++){
            if(epi[i].type==1){ cx=epi[i].pos.x; cy=epi[i].pos.y; cz=epi[i].pos.z; break; }
        }
        compact_i(epi,ni,S,true,cx,cy,cz);
        const PS::S32 na = (PS::S32)S.lst.size();
        const PS::S32 npj = compact_j_sp(spj,nj,S,cx,cy,cz,true);
        const VF e2v=vrepl(eps2), half=vrepl(0.5f), five=vrepl(5.0f), v15=vrepl(1.5f), m2=vrepl(-2.0f);
        for(PS::S32 ii=0; ii<na; ii++){
            const VF xi=vrepl(S.ix[ii]), yi=vrepl(S.iy[ii]), zi=vrepl(S.iz[ii]);
            VF ax=vzerof(), ay=vzerof(), az=vzerof(), ap=vzerof();
            for(PS::S32 j=0;j<npj;j+=VW){
                const VF dx=LARCH_VSUB(xi,LARCH_VLD(&S.jx[j]));
                const VF dy=LARCH_VSUB(yi,LARCH_VLD(&S.jy[j]));
                const VF dz=LARCH_VSUB(zi,LARCH_VLD(&S.jz[j]));
                const VF mj=LARCH_VLD(&S.jm[j]);
                const VF qxx=LARCH_VLD(&S.qxx[j]), qyy=LARCH_VLD(&S.qyy[j]), qzz=LARCH_VLD(&S.qzz[j]);
                const VF qxy=LARCH_VLD(&S.qxy[j]), qxz=LARCH_VLD(&S.qxz[j]), qyz=LARCH_VLD(&S.qyz[j]);
                VF r2=LARCH_VMADD(dx,dx,e2v);
                r2=LARCH_VMADD(dy,dy,r2);
                r2=LARCH_VMADD(dz,dz,r2);
                const VF ri=rsqrtN(r2);
                const VF ri2=LARCH_VMUL(ri,ri);
                const VF ri3=LARCH_VMUL(ri2,ri);
                const VF r5=LARCH_VMUL(LARCH_VMUL(ri2,ri3),v15);
                const VF tr=LARCH_VADD(LARCH_VADD(qxx,qyy),qzz);
                const VF qrx=LARCH_VMADD(qxz,dz,LARCH_VMADD(qxy,dy,LARCH_VMUL(qxx,dx)));
                const VF qry=LARCH_VMADD(qxy,dx,LARCH_VMADD(qyz,dz,LARCH_VMUL(qyy,dy)));
                const VF qrz=LARCH_VMADD(qyz,dy,LARCH_VMADD(qxz,dx,LARCH_VMUL(qzz,dz)));
                const VF qrr=LARCH_VMADD(qrz,dz,LARCH_VMADD(qry,dy,LARCH_VMUL(qrx,dx)));
                const VF qrr5=LARCH_VMUL(r5,qrr);
                const VF qrr7=LARCH_VMUL(ri2,qrr5);
                const VF A=LARCH_VADD(LARCH_VSUB(LARCH_VMUL(mj,ri3),LARCH_VMUL(tr,r5)),LARCH_VMUL(five,qrr7));
                const VF m2r5=LARCH_VMUL(m2,r5);
                ax=LARCH_VNMSUB(A,dx,ax); ay=LARCH_VNMSUB(A,dy,ay); az=LARCH_VNMSUB(A,dz,az);
                ax=LARCH_VNMSUB(m2r5,qrx,ax); ay=LARCH_VNMSUB(m2r5,qry,ay); az=LARCH_VNMSUB(m2r5,qrz,az);
                ap=LARCH_VSUB(ap,LARCH_VMUL(mj,ri));
                ap=LARCH_VADD(ap,LARCH_VMUL(half,LARCH_VMUL(tr,ri3)));
                ap=LARCH_VSUB(ap,qrr5);
            }
            const PS::S32 jj=S.lst[ii];
            force[jj].acc.x += G*hadd(ax);
            force[jj].acc.y += G*hadd(ay);
            force[jj].acc.z += G*hadd(az);
            force[jj].pot   += G*hadd(ap);
        }
    }
};


#endif // USE_LARCH

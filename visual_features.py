"""Perceptual hashing and feature verification for visual version grouping."""
import hashlib,time
from pathlib import Path
import cv2
import numpy as np
from scipy.fft import dctn
from PIL import Image,ImageOps

cv2.setNumThreads(1)
VERSION='phash64-256-dhash64-v1'

def bits(a):return np.packbits(a.flatten()).tobytes().hex()

def hashing(row):
    start=time.perf_counter();p=Path(row['thumbnail_path']);st=p.stat()
    with Image.open(p) as source:im=ImageOps.exif_transpose(source).convert('RGB');im.load()
    decoded=time.perf_counter();gray=im.convert('L');result={}
    for n in [8,16]:
        a=np.asarray(gray.resize((4*n,4*n),Image.Resampling.LANCZOS),dtype=float)
        coefficients=dctn(a)[:n,:n]
        result['phash'+str(n*n)]=bits(coefficients>np.median(coefficients))
    small=np.asarray(gray.resize((9,8),Image.Resampling.LANCZOS));result['dhash64']=bits(small[:,1:]>small[:,:-1])
    result.update(photo_id=row['id'],width=im.width,height=im.height,decoded_sha256=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest(),thumbnail_size=st.st_size,thumbnail_mtime_ns=st.st_mtime_ns,decode_seconds=decoded-start,compute_seconds=time.perf_counter()-decoded)
    return result

def visual(row):
    with Image.open(row['thumbnail_path']) as src:
        im=ImageOps.exif_transpose(src).convert('L');im.thumbnail((512,512),Image.Resampling.LANCZOS);a=np.asarray(im)
    points,des=cv2.SIFT_create(nfeatures=1000).detectAndCompute(a,None)
    return a,points,des

def ssim(a,b,mask):
    a=a.astype(np.float32)/255;b=b.astype(np.float32)/255
    mean_a,std_a=cv2.meanStdDev(a,mask=mask);mean_b,std_b=cv2.meanStdDev(b,mask=mask)
    gain=float(np.clip(std_a[0,0]/max(std_b[0,0],.01),.7,1.4));b=np.clip((b-mean_b[0,0])*gain+mean_a[0,0],0,1)
    blur=lambda x:cv2.GaussianBlur(x,(7,7),1.5)
    x=blur(a);y=blur(b);xx=blur(a*a)-x*x;yy=blur(b*b)-y*y;xy=blur(a*b)-x*y
    score=((2*x*y+.01**2)*(2*xy+.03**2))/((x*x+y*y+.01**2)*(xx+yy+.03**2))
    interior=cv2.erode(mask,np.ones((9,9),np.uint8))>0
    if interior.sum()<100:return 0,1
    return float(score[interior].mean()),float((score[interior]<.65).mean())

def verify(pair,features,hashes):
    a_id,b_id,d64,d256,dd=pair
    result=dict(a=a_id,b=b_id,phash64_distance=d64,phash256_distance=d256,dhash64_distance=dd,decision='rejected')
    if hashes[a_id]['decoded_sha256']==hashes[b_id]['decoded_sha256']:
        return dict(result,decision='verified_auto',reason='identical decoded pixels',ssim=1.0)
    a,ka,da=features[a_id];b,kb,db=features[b_id]
    if da is None or db is None:return dict(result,decision='candidate',reason='insufficient visual features')
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(db,da,k=2)
    good=[m for item in pairs if len(item)==2 for m,n in [item] if m.distance<.72*n.distance]
    if len(good)<12:return dict(result,reason='too few feature matches',feature_matches=len(good))
    pb=np.float32([kb[m.queryIdx].pt for m in good]);pa=np.float32([ka[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(42)
    transform,inliers=cv2.findHomography(pb,pa,cv2.RANSAC,3.0)
    if transform is None:return dict(result,reason='alignment failed')
    inside=inliers.ravel().astype(bool);n=int(inside.sum());ratio=n/len(good)
    cover_a=float(cv2.contourArea(cv2.convexHull(pa[inside]))/(a.shape[0]*a.shape[1])) if n>=3 else 0
    cover_b=float(cv2.contourArea(cv2.convexHull(pb[inside]))/(b.shape[0]*b.shape[1])) if n>=3 else 0
    warped=cv2.warpPerspective(b,transform,(a.shape[1],a.shape[0]))
    mask=cv2.warpPerspective(np.full(b.shape,255,np.uint8),transform,(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST)
    reverse=cv2.warpPerspective(np.full(a.shape,255,np.uint8),np.linalg.inv(transform),(b.shape[1],b.shape[0]),flags=cv2.INTER_NEAREST)
    overlap=min(float((mask>0).mean()),float((reverse>0).mean()))
    score,changed=ssim(a,warped,mask)
    result.update(feature_matches=len(good),inliers=n,inlier_ratio=ratio,coverage=min(cover_a,cover_b),overlap=overlap,ssim=score,changed_fraction=changed)
    accepted=n>=30 and ratio>=.6 and min(cover_a,cover_b)>=.25 and overlap>=.8 and score>=.90 and changed<=.10
    result.update(decision='verified_auto' if accepted else 'rejected',reason='high-overlap aligned visual match' if accepted else 'visual thresholds not met')
    return result

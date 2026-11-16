#!/usr/bin/env python3
"""Audit label/segmentation/adjusted-sensor semantics without fitting a model."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import scipy.io as sio
from handnet_to_tea import frame_identity, infer_split, open_source
from tea_model import DEFAULT_INTRINSICS, summary_errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",required=True,help="Prefer extracted directory for RAR data")
    p.add_argument("--output",required=True)
    p.add_argument("--max-frames",type=int,default=100)
    p.add_argument("--seed",type=int,default=20261115)
    p.add_argument("--parameters",help="HandNet Parameters.mat containing KK")
    args=p.parse_args()
    if args.max_frames<1:p.error("max-frames must bepositive")
    source=open_source(Path(args.archive).resolve())
    try:
        all_names=source.names()
        if not all_names:raise ValueError("no sourceframes")
        rng=np.random.default_rng(args.seed)
        selection=sorted(rng.choice(len(all_names),min(len(all_names),args.max_frames),replace=False).tolist())
        names=[all_names[i] for i in selection]
        if args.parameters:
            k=sio.loadmat(args.parameters)["KK"]
        else:
            v=DEFAULT_INTRINSICS;k=np.array([[v["fx"],0,v["cx"]],[0,v["fy"],v["cy"]],[0,0,1]])
        result={"schema":"handnet-label-semantics-audit-v1","source":str(Path(args.archive).resolve()),"source_frames":len(all_names),
                "sample_seed":args.seed,"frames":[],"positive_label0_inside_seg":0,"positive_label1to7_outside_seg":0,
                "positive_hand_pixels":0,"positive_background_pixels":0,"zero_depth_labeled_hand":0,"sessions":{},"positive_depth_raw_class_hist":[0]*8,
                "intrinsics":k.tolist(),"scope":"training/validation data semantics; no fittedmodel or testaccuracyclaim"}
        errors=[[] for _ in range(5)]
        for name in names:
            mat=sio.loadmat(io.BytesIO(source.read(name)),variable_names=["depth","lbl","segmap","session","idx","gidx","pos"])
            depth,label,seg=mat["depth"],mat["lbl"],mat["segmap"]>0
            if depth.shape!=label.shape or depth.shape!=seg.shape:raise ValueError("frame arrayshapesdiffer")
            result["frames"].append(frame_identity(mat,name,infer_split(args.archive)))
            for key,mask in (("positive_label0_inside_seg",(depth>0)&(label==0)&seg),
                             ("positive_label1to7_outside_seg",(depth>0)&(label>0)&~seg),
                             ("positive_hand_pixels",(depth>0)&(label>0)),
                             ("positive_background_pixels",(depth>0)&(label==0)),
                             ("zero_depth_labeled_hand",(depth==0)&(label>0))):
                result[key]+=int(mask.sum())
            result["positive_depth_raw_class_hist"]=(np.asarray(result["positive_depth_raw_class_hist"])+np.bincount(label[depth>0],minlength=8)).tolist()
            session=str(mat["session"].ravel()[0])
            result["sessions"][session]=result["sessions"].get(session,0)+1
            pos=mat["pos"][:3,:5]
            projected=k@pos
            projected=projected[:2]/projected[2]
            for class_id in range(5):
                rows,cols=np.nonzero(label==class_id+1)
                if len(rows) and np.isfinite(projected[:,class_id]).all():
                    errors[class_id].append(float(np.linalg.norm(np.array([cols.mean(),rows.mean()])-projected[:,class_id])))
        result["label_to_corresponding_pos_projection_px"]=[summary_errors(a) for a in errors]
        Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="frames"},indent=2))
    finally:
        source.close()


if __name__=="__main__":main()

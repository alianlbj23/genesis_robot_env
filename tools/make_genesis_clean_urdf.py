#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def normalize_rpy(match: re.Match) -> str:
    # 把 rpy 內容整理成 "a b c" 三個 float（缺的補 0，多的截斷）
    raw = match.group(1).strip()
    nums = [x for x in re.split(r"[,\s]+", raw) if x]
    vals = []
    for i in range(3):
        if i < len(nums):
            try:
                vals.append(str(float(nums[i])))
            except Exception:
                vals.append("0")
        else:
            vals.append("0")
    return f'rpy="{vals[0]} {vals[1]} {vals[2]}"'


def normalize_xyz(match: re.Match) -> str:
    raw = match.group(1).strip()
    nums = [x for x in re.split(r"[,\s]+", raw) if x]
    vals = []
    for i in range(3):
        if i < len(nums):
            try:
                vals.append(str(float(nums[i])))
            except Exception:
                vals.append("0")
        else:
            vals.append("0")
    return f'xyz="{vals[0]} {vals[1]} {vals[2]}"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_urdf", required=True)
    ap.add_argument("--out", dest="out_urdf", required=True)
    args = ap.parse_args()

    inp = Path(args.in_urdf)
    out = Path(args.out_urdf)
    s = inp.read_text(encoding="utf-8")

    # 1) 移除所有 inertial block（最關鍵）
    s = re.sub(r"<inertial>.*?</inertial>", "", s, flags=re.DOTALL)

    # 2) 把所有 origin 的 xyz/rpy 正規化成 3 個 float（避免奇怪格式）
    s = re.sub(r'rpy="([^"]*)"', normalize_rpy, s)
    s = re.sub(r'xyz="([^"]*)"', normalize_xyz, s)

    out.write_text(s, encoding="utf-8")
    print("Wrote:", out)

    # quick check
    has_inertial = bool(re.search(r"<\s*inertial\b", s))
    print("Has <inertial>:", has_inertial)
    # count links/joints for sanity
    print("Links:", len(re.findall(r"<\s*link\b", s)))
    print("Joints:", len(re.findall(r"<\s*joint\b", s)))


if __name__ == "__main__":
    main()

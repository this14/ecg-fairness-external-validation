r"""
patch_seed.py — train_bbb_leadcount.py 에 시드 제어를 추가한다

BSPC 리뷰어 2의 요구:
    "Conduct repeated training runs with different random seeds to establish
     the stability of the reported performance differences."

원본은 건드리지 않습니다. train_bbb_leadcount_seeded.py 를 새로 만듭니다.
원본이 그대로 남아 있어야 "기존 6개 체크포인트가 어떤 코드에서 나왔는가"를
리뷰어에게 설명할 수 있습니다.

    python patch_seed.py            # code/ 안에서 실행

바뀌는 것 (3군데)
  1. random / torch import 추가
  2. --seed 와 --deterministic 인자 추가
  3. args 파싱 직후 시드 고정 + 설정 출력

건드리지 않는 것
  데이터 로딩, 분할 인덱스, 손실 함수, 모델 구조, 학습 루프.
  시드 외에 아무것도 달라지지 않아야 기존 결과와 비교가 성립합니다.
"""
import os
import sys

SRC = "train_bbb_leadcount.py"
DST = "train_bbb_leadcount_seeded.py"

EDITS = [
    # ---- 1) import
    (
        "import os\nimport argparse\n",
        "import os\nimport argparse\nimport random\n",
    ),
    # ---- 2) 인자
    (
        '    parser.add_argument("--lr", type=float, default=1e-3)\n'
        '    args = parser.parse_args()\n',
        '    parser.add_argument("--lr", type=float, default=1e-3)\n'
        '    parser.add_argument("--seed", type=int, default=None,\n'
        '                        help="random seed; omit to reproduce the original "\n'
        '                             "unseeded behaviour of the six published checkpoints")\n'
        '    parser.add_argument("--deterministic", action="store_true",\n'
        '                        help="also force deterministic cuDNN kernels (slower, "\n'
        '                             "and some ops may raise)")\n'
        '    args = parser.parse_args()\n'
        '\n'
        '    # ---- seeding -------------------------------------------------------\n'
        '    # Only the seed changes between runs of this sweep. Data loading, the\n'
        '    # fixed split arrays, the loss, the architecture and the optimiser are\n'
        '    # untouched, so between-seed variation is attributable to initialisation\n'
        '    # and batch ordering alone.\n'
        '    if args.seed is not None:\n'
        '        random.seed(args.seed)\n'
        '        np.random.seed(args.seed)\n'
        '        torch.manual_seed(args.seed)\n'
        '        torch.cuda.manual_seed_all(args.seed)\n'
        '        os.environ["PYTHONHASHSEED"] = str(args.seed)\n'
        '        if args.deterministic:\n'
        '            torch.backends.cudnn.deterministic = True\n'
        '            torch.backends.cudnn.benchmark = False\n'
        '        print(f"[SEED] seed={args.seed} deterministic={args.deterministic}")\n'
        '    else:\n'
        '        print("[SEED] no seed set (original behaviour)")\n',
    ),
]


def main():
    if not os.path.exists(SRC):
        sys.exit(f"[중단] {SRC} 가 이 폴더에 없습니다: {os.getcwd()}\n"
                 f"       리포지토리의 code/ 폴더에서 실행하십시오.")

    with open(SRC, encoding="utf-8") as f:
        text = f.read()

    failed = []
    for i, (old, new) in enumerate(EDITS, 1):
        n = text.count(old)
        if n != 1:
            failed.append((i, n, old.splitlines()[0][:70]))
            continue
        text = text.replace(old, new, 1)

    if failed:
        print("[중단] 아래 패턴을 정확히 1회 찾지 못했습니다. 아무것도 쓰지 않았습니다.\n")
        for i, n, head in failed:
            print(f"  편집 {i}: {n}회 발견  |  {head}")
        print("\n파일이 예상과 다릅니다. train_bbb_leadcount.py 를 보내 주시면 맞추겠습니다.")
        sys.exit(1)

    header = (
        'r"""\n'
        'train_bbb_leadcount_seeded.py — patch_seed.py 가 자동 생성.\n'
        '직접 고치지 마십시오. 원본을 고친 뒤 패처를 다시 돌리십시오.\n'
        '\n'
        '원본과의 유일한 차이는 --seed 와 --deterministic 입니다.\n'
        '--seed 를 주지 않으면 원본과 동일하게 동작합니다.\n'
        '"""\n'
    )
    with open(DST, "w", encoding="utf-8") as f:
        f.write(header + text)

    print(f"[완료] {DST} 생성 ({len(EDITS)}군데 수정)")
    print("\n확인:")
    print(f"  python {DST} --help | grep -A2 seed")


if __name__ == "__main__":
    main()

"""
Download annotated chess data from Lichess Studies into data/raw/.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --users user1,user2
    python scripts/download_data.py --start-from NaSil
"""

import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Lichess Studies — known annotators (titled players / coaches with public studies)
DEFAULT_STUDY_USERS = [
    # Titled players / GMs
    "NoseKnowsAll",          # Prolific annotator — Fischer, Zurich 1953, player masterpiece series
    "Kingscrusher-YouTube",  # CM Tryfon Gavriel — massive collection, instructive games & openings
    "EricRosen",             # IM Eric Rosen — Stafford Gambit, various annotated studies
    "penguingim1",           # GM Andrew Tang
    "Craze",                 # GM Max Illingworth — Catalan, Berlin, openings
    "AbasovN",               # GM Nijat Abasov — Candidates & World Championship annotations
    "RealDavidNavara",       # GM David Navara — World Championship & major event annotations
    "lovlas",                # IM Lasse Løvik — Norway Chess, Candidates 2020
    "febloh",                # GM Felix Blohberger — King's Indian, opening ideas series
    "agileknight",           # IM Padmini Rout — Candidates & World Championship annotations
    "BrandonJacobson",       # GM Brandon Jacobson — Candidates daily annotations
    "CheckRaiseMate",        # FM Nate Solon — structured coaching studies
    "rowrulz",               # FM — Nepo-Ding 2023 World Championship
    "NaSil",                 # FM — London Chess Classic, Chess960
    "LampardFan08",          # IM — well-regarded endgame study collections
    "MashPotatoOperaGM",     # GM — multiple annotated game studies
    "thijscom",              # FM Thijs Laarhoven — Beautiful Chess Studies series
    # Prolific community educators
    "jomega",                # King & Pawn endgames, opening theory — large annotated library
    "pepellou",              # Highly acclaimed endgame studies
    "fuxia",                 # Caro-Kann, Benko Gambit, Ponziani — deeply annotated
    "Tasshaq",               # Fischer Classics, Alekhine Classics, Chess960 brilliancies
    "Bosburp",               # Sacrifice/checkmate/brilliancy series, Study Creators team founder
    "Remote_Chess_Academy",  # Large opening, strategy & annotated game library
    "johndavis_59",          # Prolific annotated game creator
    "PixelatedParcel",       # Systematic annotated game analysis
    "Mr_Penings",            # Caro-Kann, Nimzo/Bogo Indian repertoires (Lichess Staff Picks)
    "chessentialsBLOG",      # CM — French Defence, methodical opening comparisons
    "TonyRo",                # LM — regular tournament annotator
]


def _download_file(url: str, dest: Path, desc: str | None = None) -> None:
    """Stream download with resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        if r.status_code == 416:
            print(f"  already complete: {dest.name}")
            return
        r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0)) + existing
        mode = "ab" if existing else "wb"

        with open(dest, mode) as f, tqdm(
            total=total or None,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc or dest.name,
            leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=512 * 1024):
                f.write(chunk)
                bar.update(len(chunk))


def download_lichess_studies(usernames: list[str], start_from: str | None = None) -> None:
    out_dir = RAW_DIR / "lichess_studies"
    out_dir.mkdir(parents=True, exist_ok=True)

    if start_from:
        try:
            usernames = usernames[usernames.index(start_from):]
        except ValueError:
            print(f"  WARNING: --start-from user '{start_from}' not found in list, running all.")

    print(f"\n[Lichess Studies] Downloading studies for {len(usernames)} users -> {out_dir}")

    for username in usernames:
        dest = out_dir / f"{username}_studies.pgn"
        url = (
            f"https://lichess.org/api/study/by/{username}/export.pgn"
            "?comments=true&variations=true&clocks=true"
        )
        print(f"  {username} -> {dest.name}")
        try:
            _download_file(url, dest, desc=username)
        except requests.HTTPError as e:
            print(f"  ERROR {e} — skipping {username}")

        time.sleep(1)  # respect rate limit

    print("[Lichess Studies] Done.")


def main():
    parser = argparse.ArgumentParser(description="Download annotated Lichess Studies.")
    parser.add_argument(
        "--users",
        default=",".join(DEFAULT_STUDY_USERS),
        help="Comma-separated Lichess usernames.",
    )
    parser.add_argument(
        "--start-from",
        default=None,
        help="Skip all users before this username.",
    )
    args = parser.parse_args()

    usernames = [u.strip() for u in args.users.split(",") if u.strip()]
    download_lichess_studies(usernames, start_from=args.start_from)


if __name__ == "__main__":
    main()

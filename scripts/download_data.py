"""
Download annotated chess data from multiple sources into data/raw/.

Usage:
    python scripts/download_data.py                          # all sources
    python scripts/download_data.py --source beginchess
    python scripts/download_data.py --source lichess-studies --users user1,user2
"""

import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# BeginChess — annotated classic books as PGN
# ---------------------------------------------------------------------------

BEGINCHESS_BOOKS = [
    ("logical_chess_move_by_move",     "https://beginchess.com/games/lcmbm.pgn"),
    ("1001_winning_sacrifices",        "https://s3.amazonaws.com/beginchess/pgns/1001+Winning+Chess+Sacrifices+%26+Combinations.zip"),
    ("art_of_chess_analysis",          "https://s3.amazonaws.com/beginchess/pgns/art_of_chess_analysis.pgn"),
    ("art_of_positional_play",         "https://s3.amazonaws.com/beginchess/pgns/art_of_positional_play.pgn"),
    ("excelling_at_technical_chess",   "https://s3.amazonaws.com/beginchess/pgns/excelling_at_technical_chess.pgn"),
    ("informant_100_golden_games",     "https://s3.amazonaws.com/beginchess/pgns/informant_100goldengames.pgn"),
    ("larsens_good_move_guide",        "https://s3.amazonaws.com/beginchess/pgns/Larsen+-+Larsen%27s+Good+Move+Guide.pgn"),
    ("laskers_manual_of_chess",        "https://s3.amazonaws.com/beginchess/pgns/Emanuel+Lasker+-+Lasker%27s+Manual+of+Chess.pgn"),
    ("my_system",                      "https://s3.amazonaws.com/beginchess/pgns/mysystem_pgn.zip"),
    ("practical_rook_endings",         "https://s3.amazonaws.com/beginchess/pgns/Mednis+-+Practical+Rook+Endings.pgn"),
    ("secrets_of_positional_chess",    "https://s3.amazonaws.com/beginchess/pgns/secrets_of_positional_Chess.pgn"),
    ("shereshevsky_endgame_strategy",  "https://s3.amazonaws.com/beginchess/pgns/shereshevsky_endgame_strategy.pgn"),
    ("tarrasch_300_games",             "https://s3.amazonaws.com/beginchess/pgns/tarrasch_300_games.zip"),
    ("understanding_pawn_play",        "https://s3.amazonaws.com/beginchess/pgns/understandingpawnplayinchess.zip"),
    ("new_york_1924",                  "https://s3.amazonaws.com/beginchess/pgns/NewYork1924.pgn"),
]

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Source 1: BeginChess annotated books
# ---------------------------------------------------------------------------

def download_beginchess() -> None:
    out_dir = RAW_DIR / "beginchess"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[BeginChess] Downloading {len(BEGINCHESS_BOOKS)} annotated books -> {out_dir}")

    for name, url in BEGINCHESS_BOOKS:
        ext = ".zip" if url.endswith(".zip") else ".pgn"
        dest = out_dir / f"{name}{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip (exists): {dest.name}")
            continue
        print(f"  {dest.name}")
        try:
            _download_file(url, dest, desc=name)
        except requests.HTTPError as e:
            print(f"  ERROR {e} — skipping")

    print("[BeginChess] Done.")


# ---------------------------------------------------------------------------
# Source 2: Lichess Studies (human comments + evals)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download annotated chess data.")
    parser.add_argument(
        "--source",
        choices=["beginchess", "lichess-studies", "all"],
        default="all",
    )
    parser.add_argument(
        "--users",
        default=",".join(DEFAULT_STUDY_USERS),
        help="Comma-separated Lichess usernames for study download.",
    )
    parser.add_argument(
        "--start-from",
        default=None,
        help="Skip all users before this username.",
    )
    args = parser.parse_args()

    usernames = [u.strip() for u in args.users.split(",") if u.strip()]

    if args.source in ("beginchess", "all"):
        download_beginchess()
    if args.source in ("lichess-studies", "all"):
        download_lichess_studies(usernames, start_from=args.start_from)


if __name__ == "__main__":
    main()

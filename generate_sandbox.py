#!/usr/bin/env python3
"""
generate_sandbox.py  —  The Synthetic Drama Factory
30 fake internet creators. Three databases. One deeply troubled ecosystem.

Requires: Neo4j running (see .env), no OpenRouter key needed for this step.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv()

np.random.seed(42)

DB_DIR      = Path("sandbox_data")
SQLITE_PATH = DB_DIR / "creators.db"
FAISS_PATH  = DB_DIR / "drama.faiss"
META_PATH   = DB_DIR / "drama_meta.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dramapassword")


# ════════════════════════════════════════════════════════════════════
#  DOMAIN MODELS
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Creator:
    creator_id: str
    name: str
    niche: str
    follower_count: int
    bot_percentage: float
    avg_sponsorship_roi: float
    apology_video_count: int
    drama_score: float


@dataclass(frozen=True)
class Brand:
    brand_id: str
    name: str
    industry: str


# ════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA — THE 30 CREATORS
# ════════════════════════════════════════════════════════════════════
#  Columns: id, name, niche, followers, bot_pct, avg_roi, apologies, drama_score

CREATORS: List[Creator] = [
    Creator("c01", "XxNightmareZeroxX",      "gaming",                4_200_000,  0.61, -1.2, 3, 8.9),
    Creator("c02", "PixelQueenAshley",        "gaming/lifestyle",      2_800_000,  0.22,  4.7, 1, 4.1),
    Creator("c03", "DrakeWaffles",            "commentary",              890_000,  0.08,  6.1, 0, 3.2),
    Creator("c04", "VelvetThrone",            "fashion/lifestyle",    11_400_000,  0.44,  2.3, 2, 5.5),
    Creator("c05", "TechBroTyler",            "tech review",           3_100_000,  0.15,  8.9, 0, 2.1),
    Creator("c06", "CosmicKai",               "wellness/spirituality", 7_600_000,  0.73, -0.4, 5, 9.4),
    Creator("c07", "BurnerAccountBecky",      "drama/commentary",        540_000,  0.05,  3.8, 0, 6.7),
    Creator("c08", "GigachefMarcelo",         "food",                  1_200_000,  0.11,  7.2, 0, 1.8),
    Creator("c09", "AlphaGrindsetCoach",      "hustle culture",        6_800_000,  0.81,  0.3, 4, 9.1),
    Creator("c10", "SatoshiSleeper",          "crypto/finance",        2_400_000,  0.69, -3.7, 6, 9.8),
    Creator("c11", "LofiStudyWithMe_Hanako",  "study/aesthetic",       3_900_000,  0.09,  9.4, 0, 0.8),
    Creator("c12", "RagequitRodrigo",         "gaming",                1_700_000,  0.38,  2.1, 2, 5.9),
    Creator("c13", "KarmaKollector",          "reaction content",      8_200_000,  0.55,  1.6, 3, 7.3),
    Creator("c14", "ManifestMillions",        "manifestation/grift",  14_000_000,  0.77, -2.1, 7, 9.6),
    Creator("c15", "CrunchyRollDevil",        "anime commentary",        670_000,  0.12,  5.5, 1, 3.4),
    Creator("c16", "SleptOnSeason",           "music/hip-hop",           980_000,  0.25,  4.0, 1, 4.8),
    Creator("c17", "ViralViolet",             "fashion/aesthetic",    19_500_000,  0.48,  3.2, 2, 6.2),
    Creator("c18", "BasedBrunchBrigade",      "food/lifestyle",          450_000,  0.07,  8.1, 0, 1.5),
    Creator("c19", "ToxicPositivityTara",     "wellness",              5_300_000,  0.62,  1.1, 3, 7.8),
    Creator("c20", "DropkickDanielle",        "fitness",               2_100_000,  0.18,  6.8, 0, 2.9),
    Creator("c21", "LarpingLorenzo",          "RPG/LARP",                320_000,  0.03, 10.2, 0, 0.5),
    Creator("c22", "MidnightMunchkinMike",    "gaming/horror",           760_000,  0.29,  3.3, 1, 4.3),
    Creator("c23", "PseudoIntellectualPaul",  "commentary/philosophy", 1_500_000,  0.16,  5.0, 2, 5.1),
    Creator("c24", "GlitchWitch",             "tech/aesthetic",          880_000,  0.14,  7.7, 0, 2.4),
    Creator("c25", "RedFlagRosario",          "relationship advice",   4_400_000,  0.53,  0.8, 4, 8.2),
    Creator("c26", "CancelMeIfYouCan",        "drama meta",            2_900_000,  0.34,  2.9, 3, 7.1),
    Creator("c27", "BroCodeBreaker",          "commentary/drama",      1_100_000,  0.20,  4.4, 2, 5.6),
    Creator("c28", "VoidBaby",                "art/dark aesthetic",      430_000,  0.06,  9.1, 0, 1.1),
    Creator("c29", "NPC_Awakened",            "meme/commentary",       6_100_000,  0.42,  2.7, 2, 6.4),
    Creator("c30", "GriftedByGrace",          "personal finance/scam", 3_300_000,  0.75, -4.2, 8, 9.9),
]

BRANDS: List[Brand] = [
    Brand("b01", "NovaSip Energy",        "energy drinks"),
    Brand("b02", "GlowTech Skincare",     "beauty"),
    Brand("b03", "AlphaGrip Controllers", "gaming hardware"),
    Brand("b04", "ShadowVPN",             "cybersecurity"),
    Brand("b05", "MindsetMerch",          "merchandise"),
    Brand("b06", "CryptoWalletPro",       "crypto/fintech"),
    Brand("b07", "FitFuel Supplements",   "fitness/nutrition"),
    Brand("b08", "StreamGear Pro",        "streaming equipment"),
    Brand("b09", "Lumé Aesthetics",       "beauty/wellness"),
    Brand("b10", "NFTVerse",              "crypto/NFT"),
]


# ════════════════════════════════════════════════════════════════════
#  THE LORE — Leaked Discord chats, Reddit snark, PR apologies
# ════════════════════════════════════════════════════════════════════

DRAMA_CORPUS: List[Dict[str, str]] = [

    # ── XxNightmareZeroxX ────────────────────────────────────────────
    {
        "creator": "XxNightmareZeroxX", "type": "discord_leak",
        "text": (
            "[#private-org 2:47 AM] xnzx: bro if this VOD gets clipped im filing DMCA on every single person. "
            "the raid was staged lmao — the other streamer's chat was just our discord on alts. "
            "his manager owes me 12k from the collab deal and suddenly 'the receipts got lost' ok king. "
            "also DO NOT tell anyone i'm the one who reported CrunchyRollDevil's channel to YouTube. "
            "that beef is supposed to look completely organic."
        ),
    },
    {
        "creator": "XxNightmareZeroxX", "type": "reddit_snark",
        "text": (
            "r/LivestreamFail — 'XxNightmareZeroxX caught with 61% bot followers after NovaSip sponsorship audit' "
            "[3.2k upvotes, 847 comments] | Top comment: 'bro spent $40k on bots and then got a NovaSip deal "
            "worth $38k. the math isn't math-ing' | Reply: 'the apology video where he cried was filmed TWICE "
            "because he forgot to turn off his ring light and you could see his handwritten notes on the desk'"
        ),
    },
    {
        "creator": "XxNightmareZeroxX", "type": "pr_apology",
        "text": (
            "STATEMENT FROM XNZX: 'I want to address this situation with full accountability. The bot situation "
            "was a decision made without my complete understanding of how platform algorithms work. I was young, "
            "I was scared, and I trusted people I should not have trusted. [3 minutes of visible crying]. "
            "I am taking a two-week break to reflect. The NovaSip contract has absolutely nothing to do with "
            "any of this. Please do not contact their PR team.'"
        ),
    },

    # ── CosmicKai ────────────────────────────────────────────────────
    {
        "creator": "CosmicKai", "type": "discord_leak",
        "text": (
            "[#inner-circle $299/mo 11:13 PM] CosmicKai: the 'moon-charged selenite' is literally gravel from "
            "the Lowe's parking lot on Briarwood Ave. supplier charges $0.04 per piece. the 'sacred geometry "
            "activation ritual' is me holding a flashlight under my chin in my bathroom. ToxicPositivityTara "
            "is absolutely seething that I'm eating her audience demographic. I offered to collab but told her "
            "I need 30% of her GlowTech deal first. she called me spiritually bankrupt. rich coming from someone "
            "who charges $89 for a Zoom about inner child work."
        ),
    },
    {
        "creator": "CosmicKai", "type": "reddit_snark",
        "text": (
            "r/HobbyDrama — 'CosmicKai $299 moon kit is rocks from a hardware store: XRF fluorescence analysis' "
            "[8.1k upvotes] | Poster: 'Common quartzite gravel. Zero selenite mineral properties detected. "
            "Sticker applied with a standard Brother label printer, visible alignment error at 7 o'clock.' "
            "Update: Lumé Aesthetics has quietly removed every collab post. CosmicKai posted a "
            "Mercury-in-retrograde energy explanation within 6 hours of the thread going live."
        ),
    },
    {
        "creator": "CosmicKai", "type": "pr_apology",
        "text": (
            "CosmicKai Community Note: 'The universe has been asking me to slow down and deeply reflect. "
            "What I sourced was a different crystal configuration than what was advertised — my vendor "
            "misrepresented their supply chain and I am equally a victim here. I am issuing full refunds "
            "in store credit for my $497 Quantum Healing Masterclass. Please do not reach out to Lumé Aesthetics. "
            "They are family to me. This situation has absolutely nothing to do with them whatsoever.'"
        ),
    },

    # ── AlphaGrindsetCoach ───────────────────────────────────────────
    {
        "creator": "AlphaGrindsetCoach", "type": "discord_leak",
        "text": (
            "[#REAL-talk 3:21 AM] AGC_actual: mom said if I don't clean the basement by Friday she's converting "
            "it back to storage. my ENTIRE filming setup is down here. the '7-figure lifestyle' content is "
            "green-screened with an Airbnb penthouse photo from 2022. the dropshipping course screenshots are "
            "from 2019 when I accidentally sold 400 phone cases at 0.3% margin and thought I had a business. "
            "GriftedByGrace is my only real competition. I will make sure NFTVerse never talks to her. "
            "also do NOT tell anyone my legal name is Timothy."
        ),
    },
    {
        "creator": "AlphaGrindsetCoach", "type": "reddit_snark",
        "text": (
            "r/Scams — 'AlphaGrindsetCoach $1,997 course breakdown: I bought it so you don't have to' "
            "[12.4k upvotes] | Key findings: 'Module 1 is a rephrased Gary Vee blog post from 2018. "
            "Module 3 is a screenshot of a screenshot — visible JPEG artifacts at 3x zoom. "
            "The live Q&A is pre-recorded and he answers his own planted questions. "
            "His Calendly confirmation email footer reads: Sent from Timothy's iPad.'"
        ),
    },
    {
        "creator": "AlphaGrindsetCoach", "type": "pr_apology",
        "text": (
            "ALPHA STATEMENT: 'Weak men stay silent. I will not. Yes, my name is Timothy. The GREATS reframe "
            "their origin story — that is what separates them from the masses. My basement is a HEADQUARTERS. "
            "The course content is EVERGREEN, which is why it references strategies effective across all time "
            "periods. I am issuing a 48-hour refund window. Real alphas will not need it. "
            "The hate campaign against me is from men who will never see 6 figures. "
            "[No tears. One sustained direct stare into the camera for 11 seconds.]'"
        ),
    },

    # ── SatoshiSleeper ───────────────────────────────────────────────
    {
        "creator": "SatoshiSleeper", "type": "discord_leak",
        "text": (
            "[#whale-room (server now deleted) 4:02 AM] satoshi_irl: offloading 2.1M SLEEP tokens RIGHT NOW. "
            "set your limit sells above 0.0034 so the chart looks like organic movement. I'll drop the "
            "'bullish on SLEEP, this is the move fam' YouTube video in exactly 18 minutes. "
            "CryptoWalletPro still doesn't know I minted the competing SleepV2 token last month. "
            "their legal team has emailed me twice. I have not opened either email. "
            "I am not going to open either email."
        ),
    },
    {
        "creator": "SatoshiSleeper", "type": "reddit_snark",
        "text": (
            "r/CryptoCurrency — 'SatoshiSleeper on-chain forensics: SLEEP token dump documented in full' "
            "[6.7k upvotes] | Wallet 0x7f3...b92 (confirmed via ENS registry as SatoshiSleeper.eth) "
            "sold 2.1M SLEEP tokens exactly 22 minutes after his 'HOLD FOREVER' YouTube video was published. "
            "Video deleted 14 hours later. New upload title: 'Why I Left Crypto (mental health)'. "
            "CryptoWalletPro has terminated all partnerships effective immediately."
        ),
    },
    {
        "creator": "SatoshiSleeper", "type": "pr_apology",
        "text": (
            "Community: I've been struggling. The space has fundamentally changed. Those wallet transactions "
            "were executed by my financial advisor without my explicit approval and I am exploring all legal "
            "options available to me. SLEEP token was always experimental and anyone who invested should have "
            "conducted their own thorough research. I am stepping back to focus on mental health content. "
            "My next video will discuss burnout in the Web3 space. "
            "[Note: all crypto content set to private. This is unrelated to any investigation.]"
        ),
    },

    # ── ManifestMillions ─────────────────────────────────────────────
    {
        "creator": "ManifestMillions", "type": "discord_leak",
        "text": (
            "[#abundance-anchors VIP $2499/mo 1:58 AM] MM_real: the mastermind is 78% of my total gross revenue. "
            "I do not have meaningful other income streams. The 'passive income screenshot' in Module 2 is "
            "from selling the FIRST mastermind cohort. The money comes from the course about making money. "
            "I know what this is. I've made a kind of peace with it. RedFlagRosario tried to cancel me "
            "last month and I responded by recruiting 3 of her core audience members into my inner circle. "
            "Functionally I won that exchange."
        ),
    },
    {
        "creator": "ManifestMillions", "type": "reddit_snark",
        "text": (
            "r/antiMLM — 'ManifestMillions income disclosure analysis: the product is the course about selling "
            "the course' [15.8k upvotes] | 'The 97% of students who never reach 6 figures are told they "
            "didn't manifest hard enough — that their limiting beliefs are the bottleneck, not the fraud. "
            "This is a $2,499/month subscription to be systematically gaslit by a man filming in a rented "
            "Malibu kitchen. His refund policy requires a 30-day manifestation journal submission for review.'"
        ),
    },
    {
        "creator": "ManifestMillions", "type": "pr_apology",
        "text": (
            "To my abundance family: I have seen the threads. I lead with love. The framework has genuinely "
            "transformed lives — those who did not succeed carried limiting beliefs that no external program "
            "can override for them. That said, I hear you, and I am raising the mastermind price to $3,200/month "
            "to ensure only truly serious abundance-seekers can participate. Scarcity mindset will not be "
            "tolerated in this community space. Refund requests will be reviewed by our manifestation "
            "compliance team within 90 business days."
        ),
    },

    # ── GriftedByGrace ───────────────────────────────────────────────
    {
        "creator": "GriftedByGrace", "type": "discord_leak",
        "text": (
            "[#creator-ops 6:17 AM] GBG: the Financial Freedom Blueprint PDF is literally Dave Ramsey's "
            "Baby Steps with ctrl+F replace throughout. My cousin who passed the bar last year says it's "
            "'transformative enough' to be derivative rather than infringing. The $197 price point converts "
            "4x better than $47 because expensive equals credible to this demographic. NFTVerse is threatening "
            "legal action because I used their logo in my top-5 crypto scams video. I am not moving on this. "
            "They are objectively a scam and I have 847 DMs to prove it."
        ),
    },
    {
        "creator": "GriftedByGrace", "type": "reddit_snark",
        "text": (
            "r/personalfinance — 'GriftedByGrace $197 PDF is plagiarized Dave Ramsey content: full side-by-side' "
            "[21.3k upvotes] | Top comment: 'She changed Baby Step to Abundance Step and added a chapter called "
            "Manifest Your Emergency Fund. That is the entire adaptation. There is no other adaptation.' "
            "| Dave Ramsey's organization has been notified. She has simultaneously received a C&D from NFTVerse. "
            "Comment section disabled. 847 refund requests filed within 48 hours."
        ),
    },
    {
        "creator": "GriftedByGrace", "type": "pr_apology",
        "text": (
            "An Important Update: I have always been transparent about the sources that inspired my framework. "
            "Dave Ramsey's principles exist in the collective public financial consciousness and my adaptation "
            "adds a unique mindset and abundance lens that his original framework simply does not provide. "
            "All 847 refund requests will be processed within 90 business days. Regarding NFTVerse: "
            "I stand completely by my investigative content and am consulting legal counsel. "
            "[Comments disabled. DMs closed. Linktree still active and fully operational.]"
        ),
    },

    # ── ToxicPositivityTara ──────────────────────────────────────────
    {
        "creator": "ToxicPositivityTara", "type": "discord_leak",
        "text": (
            "[#wellness-warriors-pro 2:33 AM] TaraBehindTheCamera: blocked 23 accounts today for saying "
            "my 'grief is a choice' video was harmful. Those people are actively choosing to be harmed by "
            "my content and that is a mindset issue, not a content issue. GlowTech renewal is Thursday — "
            "DO NOT allow the controversy to trend in comments this week. Pay the engagement pod their "
            "$400 retainer. CosmicKai is absolutely encroaching on our entire demographic. "
            "She called ME spiritually bankrupt. I am filing this away permanently."
        ),
    },
    {
        "creator": "ToxicPositivityTara", "type": "reddit_snark",
        "text": (
            "r/QAnonCasualties — 'My sister paid ToxicPositivityTara $89 to be told depression is a mindset' "
            "[4.4k upvotes] | 'She told my sister that antidepressants were blocking her manifestation frequency. "
            "My sister quit her meds cold turkey. She's okay now but it was a genuinely dangerous two weeks "
            "and her actual psychiatrist is filing a formal complaint somewhere. "
            "GlowTech Skincare pulled their renewal the same week this thread went viral.'"
        ),
    },
    {
        "creator": "ToxicPositivityTara", "type": "pr_apology",
        "text": (
            "To my community — I have never claimed to be a licensed medical professional. My content exists "
            "for entertainment and spiritual exploration purposes only. Disclaimers are located in the video "
            "description, paragraph 4, immediately after the GlowTech Skincare promo code section. "
            "I believe deeply in the transformative power of mindset and will not retract that belief. "
            "I am sorry that some people interpreted my work in a way that caused them distress. "
            "This reflects where they were in their personal healing journey, not the intent of my content."
        ),
    },

    # ── RedFlagRosario ───────────────────────────────────────────────
    {
        "creator": "RedFlagRosario", "type": "discord_leak",
        "text": (
            "[#mod-only 12:01 AM] Rosario: quick internal audit — I give relationship advice full time and "
            "I have been on 4 dates in 3 years, one of which was a Hinge app glitch. The entire "
            "'my situationship recovery arc' series is scripted. I was in a mutually fine 3-week relationship "
            "and asked him to act more emotionally unavailable on camera for the content. He got a free dinner. "
            "BroCodeBreaker is posting that I plagiarized his red-flag framework. I did, but he lifted it from "
            "a 2019 Reddit thread, so legally we're both equally exposed. NovaSip pulled out this morning. "
            "I have until Thursday to find a new sponsor or I'm moving back to Tucson."
        ),
    },
    {
        "creator": "RedFlagRosario", "type": "reddit_snark",
        "text": (
            "r/Tinder — 'RedFlagRosario's entire situationship recovery arc was staged: the guy confirms it' "
            "[5.9k upvotes] | 'He said they dated for like 3 weeks. It ended completely fine. "
            "She called him a month later and asked if he'd be willing to act more emotionally unavailable on "
            "camera for a content series. He agreed. She made 6 videos. He got a free dinner. "
            "The BroCodeBreaker plagiarism thread is also gaining serious independent traction.'"
        ),
    },
    {
        "creator": "RedFlagRosario", "type": "pr_apology",
        "text": (
            "Hi loves — real talk: my content has always been part education, part narrative entertainment. "
            "I have been clear about this framing throughout. The BroCodeBreaker situation is a misunderstanding "
            "about shared vocabulary and cultural discourse in the dating content space. "
            "Red flags are not owned terminology. I'm taking a week off to fill my cup and come back stronger. "
            "This week's content is sponsored by FitFuel Supplements. Use code REDFLAG for 20% off your first order."
        ),
    },

    # ── KarmaKollector ───────────────────────────────────────────────
    {
        "creator": "KarmaKollector", "type": "discord_leak",
        "text": (
            "[#KK-inner 9:45 PM] KK: my reaction content structure is 73% of the original video with "
            "a 2-second commentary cut inserted every 4 minutes to establish Fair Use standing. "
            "My YouTube lawyer said 'probably fine' and I have built an entire business on probably fine. "
            "NPC_Awakened and I are in a cold war because he reacted to my reaction of his video and "
            "monetized my original commentary. I sent a DMCA notice. His lawyer replied with 9 pages. "
            "I have a sock puppet account that posts 'KK is completely unproblematic' in every drama thread. "
            "It has 2 followers. Both accounts are me."
        ),
    },
    {
        "creator": "KarmaKollector", "type": "reddit_snark",
        "text": (
            "r/youtube — 'KarmaKollector is 73% other people's content in a floating facecam: DMCA tracker' "
            "[7.2k upvotes] | Strike count: 19 filed, 17 reversed via Fair Use dispute, 2 currently pending. "
            "| Top comment: 'He once reacted to a reaction of his own reaction. The original source creator "
            "appeared for a total of 8 seconds across 34 minutes. He made $4,200 from that video. "
            "The original creator made $0 from the same content.'"
        ),
    },

    # ── VelvetThrone ─────────────────────────────────────────────────
    {
        "creator": "VelvetThrone", "type": "discord_leak",
        "text": (
            "[#brand-ops 11:27 PM] VT_mgmt: the thrift haul content is zero-effort content that "
            "outperforms actual fashion content by 3x so we are pivoting the entire channel to thrift. "
            "Problem: she has not been inside a physical thrift store. Her assistant buys everything "
            "on Depop at market price, removes the tags, she films the 'haul', items get returned after filming. "
            "Lumé Aesthetics is requesting raw engagement analytics. 44% of our engagement is from the "
            "India pod. Do not send raw data. Send the 'adjusted organic reach' numbers only."
        ),
    },
    {
        "creator": "VelvetThrone", "type": "reddit_snark",
        "text": (
            "r/blogsnark — 'VelvetThrone thrift hauls are Depop purchases filmed and returned: the investigation' "
            "[3.8k upvotes] | 'The $6 Goodwill cardigan she found was listed on Depop for $67 by a seller who "
            "recognized their own item from the hang tags visible in her background shot. "
            "The Lumé Aesthetics engagement audit thread has also dropped this week. "
            "The bot percentage figures are not flattering for a beauty partnership.'"
        ),
    },

    # ── ViralViolet ──────────────────────────────────────────────────
    {
        "creator": "ViralViolet", "type": "discord_leak",
        "text": (
            "[#management-eyes-only] INTERNAL: ViralViolet and VelvetThrone are both signed to Derek Chen "
            "of Derek Chen LLC. He signed both without disclosure, is engineering a rivalry between them "
            "to create negotiating leverage with StreamGear Pro, controls both channels' posting schedules, "
            "comment moderation teams, and holds veto over all merch drops. Both creators believe they have "
            "exclusive representation. He is billing each $18,000/month. He seated them 12 feet apart "
            "at the same brand dinner last quarter."
        ),
    },
    {
        "creator": "ViralViolet", "type": "reddit_snark",
        "text": (
            "r/Frenemies — 'ViralViolet and VelvetThrone have the exact same manager and neither knows it' "
            "[9.1k upvotes] | 'Derek Chen LLC appears in FTC-filed brand deal disclosures for both channels. "
            "He pitched StreamGear Pro with both creators on alternating slides of the same deck. "
            "He booked them at the same brand dinner last October. They were seated at opposite ends of "
            "the venue. He charged both for the event attendance as a business expense.'"
        ),
    },

    # ── LofiStudyWithMe_Hanako ───────────────────────────────────────
    {
        "creator": "LofiStudyWithMe_Hanako", "type": "discord_leak",
        "text": (
            "[#lofi-creators 4:11 AM] hanako_real: so I've confirmed that Hanako is a 27-person content studio "
            "based in Osaka. The persona does not exist as an individual human being. The cozy room is a "
            "dedicated set that gets struck and rebuilt for seasonal content. The ambient rain sounds have "
            "4 active copyright strikes from a licensed audio library. They are the most wholesome channel "
            "on the platform and it is entirely a corporate production apparatus. "
            "I am sitting with this information and I am, somehow, at complete peace with it."
        ),
    },
    {
        "creator": "LofiStudyWithMe_Hanako", "type": "reddit_snark",
        "text": (
            "r/lofi — 'LofiStudyWithMe_Hanako is a 27-person Osaka content studio: the persona does not exist' "
            "[2.1k upvotes] | Top comments: 'The content is still good though.' / "
            "'who have I been parasocially attached to for 2 years' / "
            "'The set is cozy. The cozy is real. I accept the terms.' / "
            "'I built a deeply meaningful one-sided relationship with a corporate entity and I refuse to regret it.'"
        ),
    },

    # ── NPC_Awakened ─────────────────────────────────────────────────
    {
        "creator": "NPC_Awakened", "type": "discord_leak",
        "text": (
            "[#npc-council 1:13 AM] NPC_A: the bit is that I'm an NPC who gained sentience but "
            "the real joke is that I am financially an NPC because KarmaKollector has now reacted to "
            "6 of my videos and made more money from my content than I made on the originals combined. "
            "I filed a DMCA. His lawyer sent back 9 pages of Fair Use precedent. "
            "MindsetMerch pulled the collab after the litigation noise got loud. "
            "I have $400 in savings, a character arc the algorithm loves, and a legal bill. Send help."
        ),
    },

    # ── CancelMeIfYouCan ─────────────────────────────────────────────
    {
        "creator": "CancelMeIfYouCan", "type": "discord_leak",
        "text": (
            "[#cancel-proof-bunker 3:59 AM] CMIC: I have now survived 3 cancellations. Each one grew "
            "my subscriber count by between 200k and 400k. I am the drama pipeline. "
            "If I go 90 days without public controversy my engagement drops 40%. "
            "I have begun engineering minor scandals on a quarterly production schedule. "
            "Q2's cost me $800 in burner account infrastructure and returned $22,000 in merch revenue. "
            "The 'the real me is deeply misunderstood' redemption arc is scheduled for Q3."
        ),
    },

    # ── PseudoIntellectualPaul ───────────────────────────────────────
    {
        "creator": "PseudoIntellectualPaul", "type": "discord_leak",
        "text": (
            "[#the-library 10:04 PM] PseudoPaul: my 47-minute Nietzsche deep-dive got 12,000 views. "
            "My 4-minute video where I stared at the camera and said 'sigma males don't explain themselves' "
            "got 2.8 million. I understand exactly what I have to do and I hate myself for understanding it. "
            "MindsetMerch wants a collab. I said yes. I have now betrayed every academic value I have "
            "ever publicly performed having. The sigma pipeline pays for my Foucault reading habit "
            "and I have made a specific kind of peace with that sentence."
        ),
    },

    # ── BroCodeBreaker ───────────────────────────────────────────────
    {
        "creator": "BroCodeBreaker", "type": "discord_leak",
        "text": (
            "[#receipts-vault 7:22 PM] BCB: I have documented and timestamped proof that RedFlagRosario "
            "copied my '17 dating red flags ranked by escape velocity' framework word-for-word. "
            "I also have documented proof that I adapted that framework from a 2019 Reddit thread "
            "by the user u/throwaway_heartbroke who posted it for free. "
            "I am pursuing RedFlagRosario aggressively. I am choosing not to address the Reddit attribution. "
            "This is called selective accountability and it is remarkably common in this industry."
        ),
    },

    # ── GlitchWitch ──────────────────────────────────────────────────
    {
        "creator": "GlitchWitch", "type": "discord_leak",
        "text": (
            "[#glitch-lab 8:47 PM] GW: everyone assumes I'm a self-taught techno-witch who learned "
            "to hack through sheer chaotic energy, and the reality is I have a CS degree from MIT "
            "and I'm genuinely afraid that disclosing this will collapse the entire brand mystique. "
            "I did two years of kernel-level debugging for a defense contractor. It was deeply boring. "
            "The 'chaotic magic' framing is a deliberate marketing architecture decision. "
            "StreamGear Pro wants an integration. I keep forgetting to post it. Good product though."
        ),
    },

    # ── DrakeWaffles ─────────────────────────────────────────────────
    {
        "creator": "DrakeWaffles", "type": "discord_leak",
        "text": (
            "[#waffles-warehouse 4:44 PM] DW: my entire brand is 'thoughtful guy who generates good points' "
            "and my actual production process is: identify the top-voted Reddit comment on a trending topic, "
            "spend 4 hours finding the most credible counterargument, present it as something I arrived at "
            "organically while in the shower. 6.1 ROI on sponsorships because I have zero scandal history "
            "and high perceived intellectual credibility. I am playing 4D chess by simply being consistent "
            "and normal. It is working better than anything else I have ever tried."
        ),
    },

    # ── TechBroTyler ─────────────────────────────────────────────────
    {
        "creator": "TechBroTyler", "type": "discord_leak",
        "text": (
            "[#tyler-tech 3:05 PM] TTyler: AlphaGrip sent a review unit and it is genuinely mid. "
            "The d-pad has a 3ms input lag on diagonal inputs and the triggers feel like dragging "
            "wet cardboard across sandpaper. They are paying $45,000 for the integration. "
            "I said 'the build quality has some quirks power users might notice' and then spent "
            "8 minutes praising the RGB lighting ecosystem integration. "
            "I am not proud of this. But also the RGB is actually really nice."
        ),
    },

    # ── DropkickDanielle ─────────────────────────────────────────────
    {
        "creator": "DropkickDanielle", "type": "discord_leak",
        "text": (
            "[#DD-fitness 6:58 AM] DD: FitFuel protein tastes like chalk dissolved in the concept of regret. "
            "The macro profile is mediocre at best. But FitFuel is the only supplement brand willing "
            "to work with a powerlifting-focused creator who refuses to market 'toning' content. "
            "So here we are. 6.8 ROI because I do not lie about results, only about flavor. "
            "'Chocolate brownie' is an ambitious and legally defensible application of the word brownie."
        ),
    },

    # ── SleptOnSeason ────────────────────────────────────────────────
    {
        "creator": "SleptOnSeason", "type": "reddit_snark",
        "text": (
            "r/hiphopheads — 'SleptOnSeason debut album: 4 uncleared samples, zero clearance attempts' "
            "[1.8k upvotes] | 'He believed SoundCloud freestyles existed in a legal gray zone where "
            "samples didn't require clearance because the work wasn't 'officially released.' "
            "The samples are from 3 signed artists and one active estate. "
            "MindsetMerch quietly cancelled the merch collab. The album has been fully de-listed.'"
        ),
    },

    # ── RagequitRodrigo ──────────────────────────────────────────────
    {
        "creator": "RagequitRodrigo", "type": "discord_leak",
        "text": (
            "[#RR-internal 11:52 PM] RR: the legendary controller throw rage clips are completely staged. "
            "I buy cheap controllers in bulk at $8 each from a liquidator. Real rage doesn't read well "
            "on camera — it's too still and quiet and then suddenly terrifying and the algorithm "
            "doesn't like the pacing. Staged performative rage clips hit 800k average. "
            "The one time I filmed actual genuine rage got 14,000 views and a wellness check from my mom."
        ),
    },

    # ── VoidBaby ─────────────────────────────────────────────────────
    {
        "creator": "VoidBaby", "type": "discord_leak",
        "text": (
            "[#void-dm 2:01 AM] VB: I have a 9.1 sponsorship ROI because I have zero drama, zero bots, "
            "one brand deal per calendar year with products I actually use, and I make things that are "
            "genuinely good by my own standards. The algorithm underperforms me consistently. "
            "430k followers. Every single one of them real. Every single one of them genuinely strange "
            "in the best possible way. I will never be famous. I have never been more at peace "
            "with any aspect of my existence."
        ),
    },

    # ── LarpingLorenzo ───────────────────────────────────────────────
    {
        "creator": "LarpingLorenzo", "type": "discord_leak",
        "text": (
            "[#larp-council 6:30 PM] LL: 10.2 sponsorship ROI because the LARP community will purchase "
            "anything that is sincerely recommended by someone they genuinely trust. "
            "320k followers. Every one of them real. Every one of them passionate in a specific and "
            "wonderful way. I mentioned a specific model of foam-core sword in a video and it sold out "
            "nationally in 4 hours. The mainstream creator economy does not know this niche economy "
            "exists and that invisibility is my entire competitive moat. I am protecting it."
        ),
    },

    # ── GigachefMarcelo ──────────────────────────────────────────────
    {
        "creator": "GigachefMarcelo", "type": "discord_leak",
        "text": (
            "[#kitchen-confidential 5:15 PM] GM: people assume I'm Michelin-trained because I have "
            "a French accent and once said mise en place with absolute zero hesitation. "
            "I am from Guadalajara. I completed a 6-week culinary course in 2018. "
            "Everything I know about French cuisine I learned from YouTube tutorials and Julia Child reruns. "
            "The food is genuinely good, which remains the important part. 7.2 ROI. "
            "I have never lied about a recipe and I do not intend to begin."
        ),
    },

    # ── BasedBrunchBrigade ───────────────────────────────────────────
    {
        "creator": "BasedBrunchBrigade", "type": "discord_leak",
        "text": (
            "[#brunch-ops 10:30 AM] BBB: three food bloggers who met at a pop-up market in 2021. "
            "No drama. No bot campaigns. No engagement pods. 450k followers who are there every Sunday. "
            "8.1 ROI because we only take deals for things we actually order with our own money. "
            "I think we might be the last content creators operating without compromise and it is "
            "genuinely lonelier than I expected it to be. Everyone else is playing a completely different game."
        ),
    },

    # ── CrunchyRollDevil ─────────────────────────────────────────────
    {
        "creator": "CrunchyRollDevil", "type": "discord_leak",
        "text": (
            "[#crunchydevil 8:33 PM] CRD: XxNightmareZeroxX reported my channel to YouTube because "
            "I said his Elden Ring Strength build was suboptimal in a tier list video. "
            "A 4.2 million subscriber gaming creator. Could not withstand a 670k anime channel "
            "calling his Strength/Faith hybrid mid. I had 3 videos removed and spent 6 weeks in appeal. "
            "I made a calmly worded follow-up video about the situation. He DM'd me a peace offering "
            "which was a NovaSip promotional discount code. The code was expired by 3 months."
        ),
    },

    # ── MidnightMunchkinMike ─────────────────────────────────────────
    {
        "creator": "MidnightMunchkinMike", "type": "discord_leak",
        "text": (
            "[#munchkin-hq 2:28 AM] MMM: my horror gaming content is my actual passion and the work "
            "I am most creatively proud of across everything I've made. My second channel where I "
            "review fast food drive-throughs at 2 AM is growing at 3x the rate. "
            "I don't fully know what to do with this information yet. "
            "StreamGear Pro wants an integration ad. I'm going to take it. "
            "I'll feel mildly conflicted about it and then use the revenue to fund the horror content."
        ),
    },

    # ── PixelQueenAshley ─────────────────────────────────────────────
    {
        "creator": "PixelQueenAshley", "type": "discord_leak",
        "text": (
            "[#pqa-studio 3:42 PM] PQA: lifestyle pivot ROI is sitting at 4.7 which is acceptable. "
            "The 22% bot percentage is legacy infrastructure from 2021 when I was buying followers "
            "to stay competitive in a completely saturated gaming space. I've cleaned most of it up. "
            "The gaming audience is rejecting the lifestyle content loudly in comments every week. "
            "Also: RagequitRodrigo and I dated briefly last year and I need that information "
            "to remain private on an indefinite basis. Indefinite."
        ),
    },

    # ── BurnerAccountBecky ───────────────────────────────────────────
    {
        "creator": "BurnerAccountBecky", "type": "discord_leak",
        "text": (
            "[#becky-ops 7:11 PM] BAB: I have no bot percentage because I am myself a bot-adjacent "
            "phenomenon. I run 4 burner accounts that I use to surface drama, confirm my own takes "
            "in comment sections, and occasionally start a rumor to test the ecosystem's transmission speed. "
            "The drama/commentary niche has a 3.8 ROI because the audience is deeply engaged "
            "and requires almost no production budget. I have never had a scandal because I AM "
            "the scandal infrastructure. I am the water supply."
        ),
    },
]


# ════════════════════════════════════════════════════════════════════
#  THE BEEF GRAPH — Nodes and Toxic Edges
# ════════════════════════════════════════════════════════════════════
# Format: (source, target, relationship, attributes_dict)

GRAPH_EDGES: List[Tuple] = [
    # ── HAS_BEEF_WITH ────────────────────────────────────────────────
    ("XxNightmareZeroxX",    "CrunchyRollDevil",    "HAS_BEEF_WITH", {"origin": "false DMCA over Elden Ring build criticism"}),
    ("CosmicKai",            "ToxicPositivityTara", "HAS_BEEF_WITH", {"origin": "wellness demographic overlap + GlowTech deal poaching attempt"}),
    ("AlphaGrindsetCoach",   "GriftedByGrace",      "HAS_BEEF_WITH", {"origin": "hustle-grift market territorial dispute"}),
    ("KarmaKollector",       "NPC_Awakened",        "HAS_BEEF_WITH", {"origin": "reaction channel monetizing original commentary, DMCA war"}),
    ("ManifestMillions",     "RedFlagRosario",      "HAS_BEEF_WITH", {"origin": "poached 3 core audience members into competing inner circle"}),
    ("RedFlagRosario",       "BroCodeBreaker",      "HAS_BEEF_WITH", {"origin": "plagiarized dating red-flag framework verbatim"}),
    ("VelvetThrone",         "ViralViolet",         "HAS_BEEF_WITH", {"origin": "manufactured rivalry by shared undisclosed manager Derek Chen LLC"}),
    ("SatoshiSleeper",       "CryptoWalletPro",     "HAS_BEEF_WITH", {"origin": "SLEEP token dump + secretly minted competing SleepV2 token"}),

    # ── DROPPED_BY (Creator → Brand, creator was terminated by brand) ─
    ("XxNightmareZeroxX",    "NovaSip Energy",      "DROPPED_BY",    {"reason": "61% bot follower audit failure, ROI negative"}),
    ("RedFlagRosario",       "NovaSip Energy",      "DROPPED_BY",    {"reason": "scripted situationship arc exposed publicly"}),
    ("CosmicKai",            "Lumé Aesthetics",     "DROPPED_BY",    {"reason": "selenite-as-gravel XRF forensics scandal"}),
    ("ToxicPositivityTara",  "GlowTech Skincare",   "DROPPED_BY",    {"reason": "anti-medication content, psychiatrist formal complaint"}),
    ("SatoshiSleeper",       "CryptoWalletPro",     "DROPPED_BY",    {"reason": "on-chain pump-and-dump evidence, competing token minted"}),
    ("NPC_Awakened",         "MindsetMerch",        "DROPPED_BY",    {"reason": "active DMCA litigation noise with KarmaKollector"}),
    ("SleptOnSeason",        "MindsetMerch",        "DROPPED_BY",    {"reason": "4 uncleared album samples, 3 label disputes ongoing"}),
    ("GriftedByGrace",       "NFTVerse",            "DROPPED_BY",    {"reason": "defamation claim over scam callout video"}),
    ("TechBroTyler",         "AlphaGrip Controllers","DROPPED_BY",   {"reason": "quietly dropped post-disclosure of mid hardware review negotiation"}),

    # ── SECRETLY_MANAGED_BY ──────────────────────────────────────────
    ("ViralViolet",          "Derek Chen LLC",      "SECRETLY_MANAGED_BY", {"monthly_fee_usd": 18000, "conflict": "simultaneously manages VelvetThrone without disclosure"}),
    ("VelvetThrone",         "Derek Chen LLC",      "SECRETLY_MANAGED_BY", {"monthly_fee_usd": 18000, "conflict": "simultaneously manages ViralViolet without disclosure"}),
    ("AlphaGrindsetCoach",   "Timothy's Mom",       "SECRETLY_MANAGED_BY", {"monthly_fee_usd": 0,     "note": "in-kind: free basement HQ, meals, laundry"}),
    ("ManifestMillions",     "Abundance Corp LLC",  "SECRETLY_MANAGED_BY", {"revenue_cut_pct": 40,    "note": "shell company, founder unknown, incorporated in Delaware"}),
    ("KarmaKollector",       "KK Sock Puppet Inc",  "SECRETLY_MANAGED_BY", {"note": "self-operated 4-account astroturfing operation posing as fan community"}),
]


# ════════════════════════════════════════════════════════════════════
#  DATABASE MANAGERS
# ════════════════════════════════════════════════════════════════════

class SQLiteManager:
    """Owns the math — structured creator metrics."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def setup_and_populate(self, creators: List[Creator]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS creators")
            conn.execute("""
                CREATE TABLE creators (
                    creator_id           TEXT PRIMARY KEY,
                    name                 TEXT NOT NULL,
                    niche                TEXT,
                    follower_count       INTEGER,
                    bot_percentage       REAL,
                    avg_sponsorship_roi  REAL,
                    apology_video_count  INTEGER,
                    drama_score          REAL
                )
            """)
            conn.executemany(
                "INSERT INTO creators VALUES (?,?,?,?,?,?,?,?)",
                [
                    (c.creator_id, c.name, c.niche, c.follower_count,
                     c.bot_percentage, c.avg_sponsorship_roi,
                     c.apology_video_count, c.drama_score)
                    for c in creators
                ],
            )
            conn.commit()
        print(f"  [SQLite]  {len(creators)} creators → {self.db_path}")

    def verify(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM creators").fetchone()
        print(f"  [SQLite]  Verified: {rows[0]} rows in creators table")


class FAISSManager:
    """Owns the lore — unstructured drama corpus, vector-indexed."""

    def __init__(self, faiss_path: Path, meta_path: Path) -> None:
        self.faiss_path = faiss_path
        self.meta_path  = meta_path
        print(f"  [FAISS]   Loading embedding model '{EMBED_MODEL}'...")
        self.model = SentenceTransformer(EMBED_MODEL)

    def embed_and_store(self, corpus: List[Dict[str, str]]) -> None:
        texts = [entry["text"] for entry in corpus]
        print(f"  [FAISS]   Embedding {len(texts)} drama chunks...")
        vectors = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine via inner-product index
            batch_size=32,
        ).astype(np.float32)

        dim   = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        faiss.write_index(index, str(self.faiss_path))
        with open(self.meta_path, "wb") as fh:
            pickle.dump(corpus, fh)

        print(f"  [FAISS]   {index.ntotal} vectors (dim={dim}) → {self.faiss_path}")
        print(f"  [FAISS]   Metadata                          → {self.meta_path}")

    def verify(self) -> None:
        index = faiss.read_index(str(self.faiss_path))
        with open(self.meta_path, "rb") as fh:
            meta = pickle.load(fh)
        print(f"  [FAISS]   Verified: index.ntotal={index.ntotal}, metadata entries={len(meta)}")


class GraphManager:
    """Owns the beef — directed creator/brand/entity graph stored in Neo4j."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def _wipe_sandbox(self, tx) -> None:
        tx.run("MATCH (n:SandboxNode) DETACH DELETE n")

    def _create_creator(self, tx, c: Creator) -> None:
        tx.run(
            """
            MERGE (n:SandboxNode:Creator {name: $name})
            SET n.creator_id    = $cid,
                n.niche          = $niche,
                n.follower_count = $fc,
                n.bot_pct        = $bot,
                n.drama_score    = $ds
            """,
            name=c.name, cid=c.creator_id, niche=c.niche,
            fc=c.follower_count, bot=c.bot_percentage, ds=c.drama_score,
        )

    def _create_brand(self, tx, b: Brand) -> None:
        tx.run(
            """
            MERGE (n:SandboxNode:Brand {name: $name})
            SET n.brand_id = $bid, n.industry = $industry
            """,
            name=b.name, bid=b.brand_id, industry=b.industry,
        )

    def _create_entity(self, tx, name: str) -> None:
        tx.run(
            "MERGE (n:SandboxNode:ManagementEntity {name: $name})",
            name=name,
        )

    def _create_edge(self, tx, src: str, dst: str, rel: str, attrs: Dict) -> None:
        safe_attrs = {k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool))}
        query = (
            f"MATCH (a:SandboxNode {{name: $src}}), (b:SandboxNode {{name: $dst}}) "
            f"MERGE (a)-[r:{rel}]->(b) "
            f"SET r += $attrs"
        )
        tx.run(query, src=src, dst=dst, attrs=safe_attrs)

    def build_and_save(
        self,
        creators: List[Creator],
        brands:   List[Brand],
        edges:    List[Tuple],
    ) -> None:
        management_entities = {
            "Derek Chen LLC", "Timothy's Mom",
            "Abundance Corp LLC", "KK Sock Puppet Inc",
        }

        with self.driver.session() as session:
            session.execute_write(self._wipe_sandbox)
            for c in creators:
                session.execute_write(self._create_creator, c)
            for b in brands:
                session.execute_write(self._create_brand, b)
            for entity in management_entities:
                session.execute_write(self._create_entity, entity)
            for src, dst, rel, attrs in edges:
                session.execute_write(self._create_edge, src, dst, rel, attrs)

            node_count = session.run(
                "MATCH (n:SandboxNode) RETURN count(n) AS c"
            ).single()["c"]
            edge_count = session.run(
                "MATCH (:SandboxNode)-[r]->(:SandboxNode) RETURN count(r) AS c"
            ).single()["c"]

        print(f"  [Neo4j]   {node_count} nodes, {edge_count} edges written")
        print(f"  [Neo4j]   Browser: http://localhost:7474  (neo4j / dramapassword)")

    def verify(self) -> None:
        with self.driver.session() as session:
            rows = session.run(
                "MATCH (:SandboxNode)-[r]->(:SandboxNode) "
                "RETURN type(r) AS rel, count(r) AS cnt "
                "ORDER BY cnt DESC"
            ).data()
        breakdown = {row["rel"]: row["cnt"] for row in rows}
        print(f"  [Neo4j]   Verified: edge breakdown → {breakdown}")


# ════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═" * 60)
    print("  SYNTHETIC DRAMA FACTORY  —  initializing sandbox")
    print("═" * 60 + "\n")

    DB_DIR.mkdir(exist_ok=True)

    # ── 1. SQLite ────────────────────────────────────────────────────
    print("[ 1/3 ] SQLite — The Math")
    sql = SQLiteManager(SQLITE_PATH)
    sql.setup_and_populate(CREATORS)
    sql.verify()

    # ── 2. FAISS ─────────────────────────────────────────────────────
    print("\n[ 2/3 ] FAISS  — The Lore")
    vec = FAISSManager(FAISS_PATH, META_PATH)
    vec.embed_and_store(DRAMA_CORPUS)
    vec.verify()

    # ── 3. Graph ─────────────────────────────────────────────────────
    print("\n[ 3/3 ] Neo4j  — The Beef")
    grph = GraphManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        grph.build_and_save(CREATORS, BRANDS, GRAPH_EDGES)
        grph.verify()
    finally:
        grph.close()

    print("\n" + "═" * 60)
    print("  SANDBOX FULLY INITIALIZED")
    print("═" * 60)
    print(f"  SQLite   →  {SQLITE_PATH}")
    print(f"  FAISS    →  {FAISS_PATH}")
    print(f"  Metadata →  {META_PATH}")
    print(f"  Neo4j    →  {NEO4J_URI}  (browser: http://localhost:7474)")
    print()
    print("  Next: run  python rag_engines.py  to verify retrieval.")
    print()


if __name__ == "__main__":
    main()
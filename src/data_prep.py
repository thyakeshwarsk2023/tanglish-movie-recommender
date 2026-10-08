"""
data_prep.py — Build a unified movie catalog (English + Tamil)

This script:
1. Reads raw TMDB CSV from data/raw/ for English movies
2. Includes a curated set of ~100 well-known Tamil movies with real plot summaries
3. Merges both into data/catalog.csv with a consistent schema

The curated Tamil catalog is necessary because the standard TMDB 5000 dataset
contains only 2 Tamil-language films — far too few for meaningful cross-lingual
retrieval evaluation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — works both when run directly and when imported
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.csv")


# ---------------------------------------------------------------------------
# 1. English movies from TMDB 5000
# ---------------------------------------------------------------------------
def load_english_movies(raw_csv_path: str) -> pd.DataFrame:
    """
    Load English-language movies from the TMDB 5000 movies CSV.
    Parses JSON genre strings into readable comma-separated genre names.
    """
    df = pd.read_csv(raw_csv_path)

    # Filter English-language movies
    en_df = df[df["original_language"] == "en"].copy()

    # Parse genres from JSON string to comma-separated names
    def parse_genres(genre_str):
        try:
            genres = json.loads(genre_str.replace('""', '"'))
            return ", ".join([g["name"] for g in genres])
        except (json.JSONDecodeError, TypeError, KeyError):
            return ""

    en_df["genres_parsed"] = en_df["genres"].apply(parse_genres)

    # Extract release year
    en_df["release_year"] = pd.to_datetime(
        en_df["release_date"], errors="coerce"
    ).dt.year

    # Build standardized output
    result = pd.DataFrame({
        "title": en_df["title"],
        "overview": en_df["overview"],
        "genres": en_df["genres_parsed"],
        "language": "en",
        "release_year": en_df["release_year"],
    })

    return result


# ---------------------------------------------------------------------------
# 2. Tamil movies from TMDB (if any exist in the raw CSV)
# ---------------------------------------------------------------------------
def load_tamil_from_tmdb(raw_csv_path: str) -> pd.DataFrame:
    """Extract Tamil-language entries from the same TMDB CSV."""
    df = pd.read_csv(raw_csv_path)
    ta_df = df[df["original_language"] == "ta"].copy()

    if ta_df.empty:
        return pd.DataFrame(columns=["title", "overview", "genres", "language", "release_year"])

    def parse_genres(genre_str):
        try:
            genres = json.loads(genre_str.replace('""', '"'))
            return ", ".join([g["name"] for g in genres])
        except (json.JSONDecodeError, TypeError, KeyError):
            return ""

    ta_df["genres_parsed"] = ta_df["genres"].apply(parse_genres)
    ta_df["release_year"] = pd.to_datetime(
        ta_df["release_date"], errors="coerce"
    ).dt.year

    result = pd.DataFrame({
        "title": ta_df["title"],
        "overview": ta_df["overview"],
        "genres": ta_df["genres_parsed"],
        "language": "ta",
        "release_year": ta_df["release_year"],
    })

    return result


# ---------------------------------------------------------------------------
# 3. Curated Tamil movie catalog — ~100 well-known Tamil films
#    with genuine plot summaries, genres, and release years.
#    This is necessary because TMDB 5000 only has 2 Tamil entries.
# ---------------------------------------------------------------------------
def get_curated_tamil_movies() -> pd.DataFrame:
    """
    Return a DataFrame of ~100 curated Tamil movies with real titles,
    genuine plot summaries, genres, and release years.
    """
    movies = [
        # --- Action / Thriller Classics & Hits ---
        {"title": "Baashha", "overview": "An auto driver in Madras with a humble demeanor hides a past as an underworld don in Bombay, compelled to reveal his deadly skills when criminal syndicates threaten his family.", "genres": "Action, Crime", "release_year": 1995},
        {"title": "Thalapathi", "overview": "Inspired by the Mahabharata friendship between Karna and Duryodhana, an abandoned orphan is raised in a slum and becomes the loyal right-hand enforcer of a local godfather.", "genres": "Action, Drama", "release_year": 1991},
        {"title": "Moondru Mugam", "overview": "A legendary upright police officer named Alex Pandian fights a ruthless criminal network, while years later his twin sons avenge their father's betrayal.", "genres": "Action, Crime", "release_year": 1982},
        {"title": "Captain Prabhakaran", "overview": "An elite forest officer is assigned a high-stakes mission to capture a notorious forest brigand operating in the treacherous terrain of Sathyamangalam.", "genres": "Action, Thriller", "release_year": 1991},
        {"title": "Apoorva Sagodharargal", "overview": "Twin brothers separated at birth — one a circus dwarf with mechanical genius and the other a carefree mechanic — join forces to avenge their police inspector father's murder.", "genres": "Action, Comedy, Drama", "release_year": 1989},
        {"title": "Padayappa", "overview": "An honorable engineer loses his family estate to betrayal, recovers his fortune through sheer enterprise, and faces the vengeful wrath of a spurned, aristocratic woman.", "genres": "Action, Drama", "release_year": 1999},
        {"title": "Annamalai", "overview": "A poor milkman and the son of an arrogant millionaire are best friends until deceit drives a wedge between them, turning the milkman into a formidable business tycoon.", "genres": "Action, Drama", "release_year": 1992},
        {"title": "Gentleman", "overview": "A respected kitchenware businessman moonlights as an elusive modern-day Robin Hood, stealing vast sums from corrupt elites to fund free higher education for poor students.", "genres": "Action, Thriller", "release_year": 1993},
        {"title": "Kadhalan", "overview": "A college student falls in love with the daughter of a ruthless state governor, discovering the governor's secret conspiracy to orchestrate bomb blasts and impose emergency rule.", "genres": "Action, Romance", "release_year": 1994},
        {"title": "Kuruthipunal", "overview": "Two dedicated police officers infiltrate a ruthless terrorist organization in an operation that tests their courage, family safety, and ultimate sacrifice.", "genres": "Action, Thriller", "release_year": 1995},
        {"title": "Billa", "overview": "When a notorious international smuggler dies in police custody, an innocent lookalike simpleton is recruited to impersonate him and infiltrate the gang.", "genres": "Action, Thriller", "release_year": 1980},
        {"title": "Indian", "overview": "An elderly freedom fighter takes justice into his own hands by assassinating corrupt government officials, while his own son becomes entangled in the same corruption he fights against.", "genres": "Action, Drama", "release_year": 1996},
        {"title": "Vikram", "overview": "A special agent investigates a case involving a series of murders, uncovering a drug ring operated by a masked vigilante. The story interweaves three timelines and characters in a gripping action thriller.", "genres": "Action, Thriller", "release_year": 2022},
        {"title": "Kaithi", "overview": "An ex-convict who just got released from prison tries to meet his daughter for the first time but gets caught in a war between police and drug lords in a single night of intense action.", "genres": "Action, Thriller", "release_year": 2019},
        {"title": "Master", "overview": "An alcoholic professor is sent to a juvenile home as punishment where he clashes with a gangster who uses the children for criminal activities. A battle of wits and wills ensues.", "genres": "Action, Thriller", "release_year": 2021},
        {"title": "Vada Chennai", "overview": "An innocent carrom player from North Chennai gets caught in the crossfire of two warring gangster factions, and over three decades navigates gang politics, betrayal, and survival.", "genres": "Action, Crime, Drama", "release_year": 2018},
        {"title": "Asuran", "overview": "A father and son are on the run from a powerful landlord and his henchmen after a violent confrontation over land rights, revealing the oppression and resistance of marginalized communities.", "genres": "Action, Drama", "release_year": 2019},
        {"title": "Thuppakki", "overview": "An Indian army captain on leave discovers a sleeper cell of terrorists planning attacks in Chennai and takes it upon himself to eliminate them using clever counter-intelligence tactics.", "genres": "Action, Thriller", "release_year": 2012},
        {"title": "Kaththi", "overview": "A lookalike criminal escapes from a Malaysian prison and assumes the identity of a social activist, eventually discovering the plight of farmers fighting against a corporate giant over water rights.", "genres": "Action, Drama", "release_year": 2014},
        {"title": "Mankatha", "overview": "A suspended cop plans a heist targeting cricket betting money, assembling a team of criminals. Double-crosses and betrayals unfold as everyone tries to grab the loot for themselves.", "genres": "Action, Thriller, Crime", "release_year": 2011},
        {"title": "Vishwaroopam", "overview": "A Kathak dance teacher is revealed to be an undercover RAW agent who infiltrated an Afghan terrorist organization. When the plot targets a nuclear attack on New York, he must act.", "genres": "Action, Thriller", "release_year": 2013},
        {"title": "Ghajini", "overview": "A businessman suffering from short-term memory loss tattoos clues on his body to hunt down the man who murdered his girlfriend, in a non-linear thriller about revenge and memory.", "genres": "Action, Thriller", "release_year": 2005},
        {"title": "Kaakha Kaakha", "overview": "A dedicated police officer hunts a serial killer in Chennai while navigating a new romance and the challenges of police life, leading to a devastating personal tragedy.", "genres": "Action, Thriller, Romance", "release_year": 2003},
        {"title": "Vettaiyaadu Vilaiyaadu", "overview": "A tough cop travels from Chennai to New York to track down a pair of serial killers targeting young women, in a gritty cat-and-mouse thriller.", "genres": "Action, Thriller, Crime", "release_year": 2006},
        {"title": "Mersal", "overview": "A magician with a troubled past takes on the corrupt healthcare system and a powerful politician to avenge his father's death, revealing hidden connections between three lookalikes.", "genres": "Action, Drama, Thriller", "release_year": 2017},
        {"title": "Sarkar", "overview": "A successful NRI businessman returns to India to vote and discovers massive electoral fraud, leading him to take on the corrupt political establishment to clean up the system.", "genres": "Action, Drama, Thriller", "release_year": 2018},
        {"title": "Beast", "overview": "A RAW agent suffering from PTSD is trapped inside a shopping mall during a terrorist hijacking and must rescue the hostages while battling his inner demons.", "genres": "Action, Thriller", "release_year": 2022},
        {"title": "Jailer", "overview": "A retired jailer's peaceful life is disrupted when his police officer son is attacked by a powerful crime lord. He comes out of retirement to bring down the criminal empire.", "genres": "Action, Drama", "release_year": 2023},
        {"title": "Leo", "overview": "A mild-mannered cafe owner with a hidden violent past is forced to confront his former life when a drug cartel threatens his family and peaceful existence.", "genres": "Action, Thriller", "release_year": 2023},
        {"title": "Vikram Vedha", "overview": "A tough cop's moral certainty is shaken when a notorious gangster surrenders and tells him three stories that challenge his understanding of right and wrong.", "genres": "Action, Thriller, Crime", "release_year": 2017},
        {"title": "Yennai Arindhaal", "overview": "An undercover cop builds a peaceful family life after a dangerous assignment, only to be drawn back into conflict when a vengeful gangster resurfaces.", "genres": "Action, Thriller", "release_year": 2015},

        # --- Iconic Comedies ---
        {"title": "Michael Madana Kama Rajan", "overview": "Four quadruplet brothers separated at infancy — a wealthy heir, a fiery cook, an adept thief, and an eccentric farmer — get entangled in hilarious mistaken identity mayhem.", "genres": "Comedy", "release_year": 1990},
        {"title": "Panchatanthiram", "overview": "A commercial pilot and his four loyal buddies try to conceal a dead escort's body during a wild trip, setting off an escalating series of comedic blunders and misunderstandings.", "genres": "Comedy", "release_year": 2002},
        {"title": "Soodhu Kavvum", "overview": "A gang of quirky kidnappers who only target corrupt politicians find themselves in over their heads when they kidnap a minister's son who turns out to have orchestrated his own kidnapping.", "genres": "Comedy, Crime", "release_year": 2013},
        {"title": "Boss Engira Bhaskaran", "overview": "A carefree, unemployed youth starts a tutorial college with his quirky friend to prove his worth to the family of the schoolteacher he loves.", "genres": "Comedy, Romance", "release_year": 2010},
        {"title": "Ullathai Allitha", "overview": "A wealthy runaway switches identities with a street-smart driver, leading to nonstop farcical romance and chaotic misunderstandings in the Nilgiris.", "genres": "Comedy, Romance", "release_year": 1996},
        {"title": "Kalakalappu", "overview": "Two brothers struggle to keep their ancestral, debt-ridden traditional cafe running while getting tangled in a diamond heist and local gangster turf wars.", "genres": "Comedy", "release_year": 2012},
        {"title": "Vasool Raja MBBS", "overview": "A lovable local extortionist pretends to be a medical student to fulfill his father's dream, bringing laughter, empathy, and unorthodox healing methods to the hospital.", "genres": "Comedy, Drama", "release_year": 2004},
        {"title": "Pammal K. Sambandam", "overview": "A confirmed bachelor who despises women clashes with a fierce feminist doctor, sparking a witty battle of the sexes after a misplaced surgical watch.", "genres": "Comedy, Romance", "release_year": 2002},
        {"title": "Thenali", "overview": "A Sri Lankan Tamil refugee burdened with bizarre phobias of everything is sent to a vacationing psychiatrist, inadvertently turning the doctor's life upside down with hilarious paranoia.", "genres": "Comedy", "release_year": 2000},
        {"title": "Avvai Shanmugi", "overview": "A devoted divorced father transforms himself into an elderly Brahmin nanny to stay close to his beloved daughter in his ex-wife's aristocratic household.", "genres": "Comedy, Drama", "release_year": 1996},
        {"title": "Kadhalikka Neramillai", "overview": "Three friends concoct elaborate deceptions and wild alter-egos to win the approval of an aristocratic, strict patriarch in this timeless romantic comedy classic.", "genres": "Comedy, Romance", "release_year": 1964},
        {"title": "Sathi Leelavathi", "overview": "When a husband begins an extramarital affair with his glamorous colleague, his quirky wife and loyal friends conspire in a hilarious scheme to win him back.", "genres": "Comedy", "release_year": 1995},
        {"title": "Karakattakkaran", "overview": "Two rival traditional Karakattam folk dancers fall in love amidst rural village festivals, legendary comedy encounters over bananas, and comedic family feuds.", "genres": "Comedy, Romance", "release_year": 1989},
        {"title": "Aahaa", "overview": "A lively joint family's cheerful unity is tested by an alleged secret love child, prompting the younger brother to secretly resolve the crisis without breaking family harmony.", "genres": "Comedy, Family, Drama", "release_year": 1997},
        {"title": "Oru Kal Oru Kannadi", "overview": "An easygoing youth and his sarcastic best friend bumble through courtship, quirky job interviews, and ridiculous misadventures across Chennai.", "genres": "Comedy, Romance", "release_year": 2012},
        {"title": "Varuthapadatha Valibar Sangam", "overview": "A lazy but charming village youth tries to win the heart of a new girl in town while dealing with a rival suitor and the comedic chaos of rural life.", "genres": "Comedy, Romance", "release_year": 2013},
        {"title": "Naduvula Konjam Pakkatha Kaanom", "overview": "A man loses his recent memory just days before his wedding, leading to hilarious situations as his friends desperately try to help him remember while keeping the secret from everyone.", "genres": "Comedy, Drama", "release_year": 2012},
        {"title": "Friends", "overview": "Three inseparable college friends navigate romance, rivalries, and coming-of-age challenges in this feel-good comedy about friendship and youth.", "genres": "Comedy, Drama, Romance", "release_year": 2001},
        {"title": "Winner", "overview": "A college student with extraordinary luck keeps winning at everything he tries, attracting trouble and jealousy in this lighthearted comedy with iconic comic banter.", "genres": "Comedy", "release_year": 2003},
        {"title": "Idharkuthane Aasaipattai Balakumara", "overview": "Multiple storylines intersect in this dark comedy about love, lust, and liquor in Chennai, featuring colorful characters whose paths cross in unexpected ways.", "genres": "Comedy", "release_year": 2013},
        {"title": "Jigarthanda", "overview": "An aspiring filmmaker decides to make a gangster movie and begins following a real-life gangster for research, leading to a dangerous and darkly comic entanglement.", "genres": "Comedy, Crime, Thriller", "release_year": 2014},
        {"title": "Good Night", "overview": "A relatable young IT employee with an uncontrollable snoring condition marries the woman he loves, facing comedic and emotional turbulence as it strains their marriage.", "genres": "Comedy, Drama, Romance", "release_year": 2023},
        {"title": "Aandavan Kattalai", "overview": "A rural youth travels to Chennai seeking a passport with a fake spouse name to work in London, discovering the absurd bureaucratic traps of the immigration underworld.", "genres": "Comedy, Drama", "release_year": 2016},
        {"title": "Mandela", "overview": "A barber in a remote village becomes the deciding vote in a local election and discovers his unexpected political power when both parties try to win him over.", "genres": "Comedy, Drama", "release_year": 2021},

        # --- Sci-Fi & Time Travel ---
        {"title": "Enthiran", "overview": "A scientist creates a humanoid robot with artificial intelligence that begins to develop human emotions and falls in love with the scientist's girlfriend, leading to chaos when the robot turns evil.", "genres": "Science Fiction, Action", "release_year": 2010},
        {"title": "2.0", "overview": "When mobile phones across the city start disappearing and a mysterious bird-shaped entity threatens humanity, a scientist must reactivate his robot creation to fight this new menace.", "genres": "Science Fiction, Action", "release_year": 2018},
        {"title": "24", "overview": "A scientist invents a time-traveling watch that becomes the center of conflict between his son and an evil villain, both trying to control the power of time manipulation.", "genres": "Science Fiction, Action, Thriller", "release_year": 2016},
        {"title": "Indru Netru Naalai", "overview": "Two friends discover a time machine and use it for personal gain, but their actions create dangerous ripple effects that threaten to destroy everything they know.", "genres": "Science Fiction, Comedy", "release_year": 2015},
        {"title": "Maanaadu", "overview": "A common man attending a political rally in Coimbatore gets caught in a deadly time loop, forced to repeatedly relive the day to prevent the assassination of the Chief Minister.", "genres": "Science Fiction, Action, Thriller", "release_year": 2021},
        {"title": "Tik Tik Tik", "overview": "A magician and escape artist is recruited by the defense space division for a daring mission into outer space to divert an asteroid hurtling towards Chennai.", "genres": "Science Fiction, Action, Adventure", "release_year": 2018},
        {"title": "7aum Arivu", "overview": "A genetic engineer attempts to reactivate the dormant skills and martial arts mastery of the ancient 6th-century monk Bodhidharma in his modern descendant to stop biological warfare.", "genres": "Science Fiction, Action", "release_year": 2011},
        {"title": "Mark Antony", "overview": "Two gangster sons discover a retro telephone that allows them to communicate with the past, using time alterations to change their fathers' fates.", "genres": "Science Fiction, Comedy, Action", "release_year": 2023},

        # --- Romance & Heartwarming ---
        {"title": "96", "overview": "Two high school sweethearts meet after 22 years at a school reunion and spend one emotional night recounting memories, unfulfilled love, and what could have been.", "genres": "Romance, Drama", "release_year": 2018},
        {"title": "Alaipayuthey", "overview": "Two young people fall in love and marry against their families' wishes, only to discover that the challenges of daily married life are harder than the fight to be together.", "genres": "Romance, Drama", "release_year": 2000},
        {"title": "Mouna Ragam", "overview": "A young woman forced into an arranged marriage struggles to accept her new husband while haunted by memories of a past relationship, gradually finding common ground.", "genres": "Romance, Drama", "release_year": 1986},
        {"title": "Roja", "overview": "A woman's husband, a cryptographer, is kidnapped by militants in Kashmir. She embarks on a desperate journey to secure his release, becoming a symbol of love and national unity.", "genres": "Romance, Drama, Thriller", "release_year": 1992},
        {"title": "Bombay", "overview": "An inter-religious love story set against the backdrop of the 1992-93 Bombay riots, showing how a Hindu-Muslim couple and their twin sons survive communal violence.", "genres": "Romance, Drama", "release_year": 1995},
        {"title": "Vinnaithaandi Varuvaayaa", "overview": "A young Tamil Hindu aspiring filmmaker falls in love with a Malayali Christian girl in the streets of Chennai, navigating cultural and religious barriers to their love.", "genres": "Romance, Drama", "release_year": 2010},
        {"title": "Minnale", "overview": "A young engineering graduate falls in love with a woman at a wedding and pursues her to her college, only to discover she is engaged to someone else, setting up a love triangle.", "genres": "Romance, Comedy", "release_year": 2001},
        {"title": "Kandukondain Kandukondain", "overview": "Inspired by Sense and Sensibility, three sisters face financial hardship, unrequited romance, and career ambitions in Chennai while staying true to their dignity.", "genres": "Romance, Drama", "release_year": 2000},
        {"title": "Rhythm", "overview": "A widowed photojournalist and an editor connect through their shared past traumas and music, gently navigating emotional healing and second chances at love.", "genres": "Romance, Drama", "release_year": 2000},
        {"title": "Kaadhal Kottai", "overview": "Two people fall deeply in love through epistolary letters without ever having seen each other's faces, navigating years of near misses in Chennai.", "genres": "Romance, Drama", "release_year": 1996},
        {"title": "Thulladha Manamum Thullum", "overview": "An aspiring singer unintentionally causes a girl to lose her eyesight, secretly dedicating his life to funding her cornea surgery while concealing his identity from her.", "genres": "Romance, Drama", "release_year": 1999},
        {"title": "Chinna Thambi", "overview": "An innocent rural folk singer forms a pure bond with a wealthy, overprotected girl, defying the wrath of her tyrannical, aristocratic elder brothers.", "genres": "Romance, Drama", "release_year": 1991},
        {"title": "Moondram Pirai", "overview": "A schoolteacher rescues an amnesiac woman who has regressed to childhood innocence from a brothel and nurtures her in Ooty, facing heartbreak when her memory suddenly returns.", "genres": "Romance, Drama", "release_year": 1982},
        {"title": "OK Kanmani", "overview": "Two commitment-phobic young professionals in Mumbai decide to live together without marriage, but gradually develop deep feelings while caring for their elderly landlord.", "genres": "Romance, Drama", "release_year": 2015},
        {"title": "Sillunu Oru Kaadhal", "overview": "A married man's peaceful family life in Mumbai is shaken when his wife discovers his intense college diary and his passionate first love from years ago.", "genres": "Romance, Drama", "release_year": 2006},
        {"title": "Irugapatru", "overview": "Three married couples with differing emotional temperaments seek marriage counseling, exploring the vulnerabilities and reconciliation required for long-term love.", "genres": "Romance, Drama", "release_year": 2023},
        {"title": "Pyaar Prema Kaadhal", "overview": "An innocent, traditional middle-class boy falls in love with an independent, career-driven modern woman, navigating conflicting perspectives on relationships and cohabitation.", "genres": "Romance, Comedy", "release_year": 2018},

        # --- Emotional Dramas & Masterpieces ---
        {"title": "Nayakan", "overview": "Inspired by The Godfather, a slum dweller in Bombay rises to become the don of Dharavi after witnessing his father's murder, navigating moral complexities of power and justice.", "genres": "Drama, Crime", "release_year": 1987},
        {"title": "Thevar Magan", "overview": "An MBA graduate returns from London to his village and must navigate feudal traditions, family honor, and caste politics after his father's murder.", "genres": "Drama", "release_year": 1992},
        {"title": "Kannathil Muthamittal", "overview": "An adopted girl sets out to find her biological mother in war-torn Sri Lanka, leading her adoptive parents on an emotional journey into the LTTE conflict zone.", "genres": "Drama, War", "release_year": 2002},
        {"title": "Anbe Sivam", "overview": "Two strangers with opposing worldviews — a selfish ad filmmaker and a disfigured communist social worker — are stranded together during a journey, discovering compassion and humanity.", "genres": "Drama, Comedy", "release_year": 2003},
        {"title": "Autograph", "overview": "On the eve of his wedding, a young man revisits the significant women in his life — his mother, first love, college friend, and mentor — through a nostalgic journey.", "genres": "Drama, Romance", "release_year": 2004},
        {"title": "Mahanadhi", "overview": "An honest widower is swindled into financial ruin and framed in an investment scam, leading to prison and a harrowing quest to rescue his kidnapped daughter from sex trafficking.", "genres": "Drama, Crime", "release_year": 1994},
        {"title": "Sethu", "overview": "A hot-headed college ruffian falls genuinely in love with a timid Brahmin girl, but after surviving a brutal attack by thugs, is sent to a mental asylum with tragic consequences.", "genres": "Drama, Romance", "release_year": 1999},
        {"title": "Pithamagan", "overview": "A man who grew up in a crematorium befriends a petty thief, and together they navigate society's rejection and find an unlikely brotherhood in the margins of civilization.", "genres": "Drama", "release_year": 2003},
        {"title": "Vaaranam Aayiram", "overview": "A son narrates his father's life story and their beautiful bond while searching for his father who has gone missing, interweaving two timelines of love, loss, and family.", "genres": "Drama, Romance", "release_year": 2008},
        {"title": "Deiva Thirumagal", "overview": "A mentally challenged father fights a legal battle for the custody of his young daughter, proving that love needs no intellectual capacity.", "genres": "Drama, Family", "release_year": 2011},
        {"title": "Kaaka Muttai", "overview": "Two slum children in Chennai dream of tasting pizza from a newly opened pizzeria near their home and embark on a mission to earn enough money to buy one.", "genres": "Drama, Family", "release_year": 2014},
        {"title": "Thanga Meenkal", "overview": "A struggling single father tries to give his young daughter a happy childhood while dealing with poverty, showcasing the unbreakable bond between parent and child.", "genres": "Drama, Family", "release_year": 2013},
        {"title": "Peranbu", "overview": "A single father raises his daughter with cerebral palsy, discovering the meaning of unconditional love through their challenges, encounters with different people, and personal growth.", "genres": "Drama", "release_year": 2018},
        {"title": "Pariyerum Perumal", "overview": "A law student from a marginalized community faces systemic caste discrimination at college while developing a friendship with an upper-caste classmate, exposing deep social inequalities.", "genres": "Drama", "release_year": 2018},
        {"title": "Karnan", "overview": "Inspired by real events, a fearless young man leads his oppressed village community in a revolt against systemic caste discrimination and police brutality.", "genres": "Drama, Action", "release_year": 2021},
        {"title": "Jai Bhim", "overview": "Based on a true story, a lawyer fights for justice for a tribal man who goes missing after being arrested by the police on false charges, exposing custodial violence and caste oppression.", "genres": "Drama, Legal Thriller", "release_year": 2021},
        {"title": "Soorarai Pottru", "overview": "Inspired by the life of Air Deccan founder, a young man from a small village dreams of starting an affordable airline to make flying accessible to the common man, battling powerful aviation cartels.", "genres": "Drama, Biography", "release_year": 2020},
        {"title": "Kadaisi Vivasayi", "overview": "An 85-year-old farmer, the last remaining farmer in a village turning to real estate development, becomes the prime suspect when dead bodies start appearing in his fields.", "genres": "Drama, Mystery", "release_year": 2022},
        {"title": "Kizhakku Cheemayile", "overview": "A touching rural story of the deep, unbreakable bond between an affectionate brother and sister, and how matrimonial misunderstandings tear their lives apart.", "genres": "Drama, Family", "release_year": 1993},
        {"title": "Dada", "overview": "An immature college student is suddenly forced into single fatherhood when his partner leaves their newborn baby, learning responsibility and unconditional love.", "genres": "Drama, Family, Comedy", "release_year": 2023},
        {"title": "Chithha", "overview": "A devoted uncle's idyllic life raising his young niece is shattered when the child goes missing in a small town terrorized by a child predator.", "genres": "Drama, Thriller", "release_year": 2023},
        {"title": "Gargi", "overview": "A humble schoolteacher wages an uphill legal and emotional crusade to prove her elderly father's innocence when he is accused of a heinous crime.", "genres": "Drama, Legal Thriller", "release_year": 2022},
        {"title": "Aruvi", "overview": "A spirited young woman facing social ostracization due to an illness takes hostage a sensationalist reality TV show to confront the hypocrisy of society.", "genres": "Drama, Thriller", "release_year": 2017},
        {"title": "Veyil", "overview": "A prodigal son who ran away from home after a brutal thrashing returns to his impoverished family years later, seeking forgiveness and emotional closure.", "genres": "Drama, Family", "release_year": 2006},
        {"title": "Kattradhu Thamizh", "overview": "A passionate Tamil literature graduate finds himself alienated and marginalized in a hyper-capitalist city dominated by corporate IT wealth, spiraling into despondency.", "genres": "Drama, Crime", "release_year": 2007},
        {"title": "Iruvar", "overview": "Inspired by the real-life friendship and political rivalry between MGR and Karunanidhi, two friends in Tamil cinema rise to power and eventually become bitter political opponents.", "genres": "Drama, Biography, Romance", "release_year": 1997},
        {"title": "Sarpatta Parambarai", "overview": "Set in 1970s Madras, a young man from a lower-class boxing clan gets a chance to fight for glory and honor against a rival clan, battling personal demons and social barriers.", "genres": "Drama, Sports", "release_year": 2021},
        {"title": "Super Deluxe", "overview": "Four interconnected stories unfold in Chennai: a transgender woman returning home, a couple dealing with an unexpected death, a priest questioning faith, and a boy discovering his mother's secret.", "genres": "Drama, Comedy, Thriller", "release_year": 2019},
        {"title": "Virumandi", "overview": "A murder case in a village is told from two opposing perspectives — the accused and a witness — exploring justice, truth, and the death penalty through Rashomon-style storytelling.", "genres": "Drama, Crime", "release_year": 2004},

        # --- Mystery & Thriller ---
        {"title": "Ratsasan", "overview": "An aspiring filmmaker turned reluctant cop hunts a serial killer who targets schoolgirls, and the investigation becomes personal when the killer gets closer to his own family.", "genres": "Thriller, Mystery", "release_year": 2018},
        {"title": "Sigappu Rojakkal", "overview": "A charming psychopath methodically targets women in this pioneering Tamil psychological thriller that shocked audiences with its dark exploration of a serial killer's mind.", "genres": "Thriller, Mystery", "release_year": 1978},
        {"title": "Dhuruvangal Pathinaaru", "overview": "A retired police officer recounts his most challenging case — a series of connected murders that led him through a web of deception with a shocking twist ending.", "genres": "Mystery, Thriller", "release_year": 2016},
        {"title": "Thadam", "overview": "A murder case is complicated when the police discover two identical suspects who are not related, leading to a mind-bending investigation about identity and deception.", "genres": "Mystery, Thriller", "release_year": 2019},
        {"title": "Anniyan", "overview": "A frustrated common man with multiple personality disorder transforms into a vigilante who punishes corrupt citizens, switching between three distinct personalities.", "genres": "Thriller, Drama, Action", "release_year": 2005},
        {"title": "Imaikkaa Nodigal", "overview": "A CBI officer races against time to catch a brilliant psychopath who kidnaps people and forces their loved ones to commit crimes, leading to a battle of minds.", "genres": "Thriller, Mystery, Action", "release_year": 2018},
        {"title": "Theeran Adhigaaram Ondru", "overview": "Based on real events, a determined police officer tracks a gang of nomadic criminals who commit robberies and murders across state borders in a years-long investigation.", "genres": "Thriller, Action, Crime", "release_year": 2017},

        # --- Horror & Supernatural ---
        {"title": "Aranmanai", "overview": "A family returns to their ancestral palace for a wedding and discovers it is haunted by a vengeful spirit with a dark connection to the family's past.", "genres": "Horror, Comedy", "release_year": 2014},
        {"title": "Maya", "overview": "A woman moves into a new apartment and begins experiencing terrifying supernatural events that connect to the previous tenant's mysterious disappearance.", "genres": "Horror, Thriller", "release_year": 2015},
        {"title": "Pisasu", "overview": "A violinist begins to sense the presence of a young woman's ghost in his apartment after a fatal accident, developing an unusual and emotional bond with the spirit.", "genres": "Horror, Drama", "release_year": 2014},
        {"title": "Demonte Colony", "overview": "Four friends enter an abandoned colonial bungalow on a dare and awaken something dark and terrifying that follows them home, refusing to let them escape.", "genres": "Horror, Thriller", "release_year": 2015},
        {"title": "Aval", "overview": "A neurosurgeon and his wife move into a new house next to an old woman who dies mysteriously. Strange occurrences lead them to uncover a terrifying supernatural presence.", "genres": "Horror, Thriller", "release_year": 2017},
        {"title": "Kanchana", "overview": "A cowardly man who is afraid of everything gets possessed by the ghost of a transgender woman seeking revenge against those who murdered her, leading to horror-comedy mayhem.", "genres": "Horror, Comedy", "release_year": 2011},
        {"title": "Pizza", "overview": "A pizza delivery boy delivers to a haunted bungalow and must survive a night of supernatural horror, but nothing is what it seems in this twist-filled thriller.", "genres": "Horror, Thriller", "release_year": 2012},
        {"title": "Yaamirukka Bayamey", "overview": "A group of friends staying in a resort experience unexplained scary events that may or may not be supernatural, in this comedy horror with multiple twists.", "genres": "Horror, Comedy", "release_year": 2014},
        {"title": "Iruttu Araiyil Murattu Kuthu", "overview": "A group of youngsters spending a night in a supposedly haunted house encounter bizarre and comedic supernatural events.", "genres": "Horror, Comedy", "release_year": 2018},

        # --- Historical & Period ---
        {"title": "Ponniyin Selvan: I", "overview": "Set in the 10th century Chola dynasty, multiple factions plot to seize the throne while a young warrior embarks on a dangerous mission to warn the crown prince of a conspiracy.", "genres": "Historical, Drama, Action", "release_year": 2022},
        {"title": "Ponniyin Selvan: II", "overview": "The epic conclusion of the Chola dynasty saga as alliances shift, betrayals deepen, and the fate of the empire hangs in the balance of a final conflict.", "genres": "Historical, Drama, Action", "release_year": 2023},
        {"title": "Aayirathil Oruvan", "overview": "An army expedition searching for a lost Chola treasure in the jungles of Southeast Asia encounters a hidden civilization and ancient secrets that defy explanation.", "genres": "Action, Adventure, Historical", "release_year": 2010},
        {"title": "Madras", "overview": "Two friends in the working-class neighborhood of North Madras get caught up in local political rivalries painted on walls, leading to violence and betrayal.", "genres": "Drama, Action", "release_year": 2014},
        {"title": "Irudhi Suttru", "overview": "A disgraced boxing coach discovers raw talent in a young fisherwoman and trains her for a national championship, battling corruption and personal demons along the way.", "genres": "Drama, Sports", "release_year": 2016},

        # --- Vintage & 80s/90s Classics ---
        {"title": "Murattu Kaalai", "overview": "An honorable, humble village milkman and bull tamer stands up against an arrogant feudal landlord who schemes to seize the villagers' lands.", "genres": "Action, Drama", "release_year": 1980},
        {"title": "Sakalakala Vallavan", "overview": "A clever rural youth transforms into an educated, sophisticated gentleman to teach an arrogant wealthy woman and her haughty family a lesson in humility.", "genres": "Action, Comedy, Romance", "release_year": 1982},
        {"title": "Payanangal Mudivathillai", "overview": "A struggling stage singer finds meteoric stardom through the encouragement of a wealthy woman who loves him, facing tragedy when illness threatens their romance.", "genres": "Romance, Drama", "release_year": 1982},
        {"title": "Vaidehi Kathirunthal", "overview": "A heartbroken village wanderer with a tragic past helps a young widow find solace and protects her from cruel village gossip through music and sacrifice.", "genres": "Romance, Drama", "release_year": 1984},
        {"title": "Sindhu Bhairavi", "overview": "A celebrated Carnatic music maestro battling alcoholism and artistic stagnation is inspired by an intellectual young music lover, creating profound emotional upheaval with his devoted wife.", "genres": "Drama, Romance", "release_year": 1985},
        {"title": "Punnagai Mannan", "overview": "A despondent lover survives a suicide pact that claimed his girlfriend's life and opens a dance school, gradually learning to embrace life when a vivacious student falls for him.", "genres": "Romance, Drama", "release_year": 1986},
        {"title": "Agni Natchathiram", "overview": "Two estranged half-brothers with bitter animosity toward each other are forced to unite to protect their father when an assassination plot threatens his life.", "genres": "Action, Drama", "release_year": 1988},
        {"title": "Idhu Namma Aalu", "overview": "A charming middle-class youth uses witty deceptions and humor to court a traditional Brahmin girl, navigating hilarious cultural clashes with her conservative family.", "genres": "Comedy, Romance", "release_year": 1988},
        {"title": "Gopura Vasalile", "overview": "A carefree rural schoolteacher falls in love with a mysterious, introverted woman who comes to live in his village with a hidden past.", "genres": "Comedy, Romance, Drama", "release_year": 1991},
        {"title": "Singaravelan", "overview": "A simple village youth journeys to the bustling city of Madras to fulfill his mother's wish of finding and marrying his estranged wealthy cousin.", "genres": "Comedy, Romance", "release_year": 1992},
        {"title": "Chinna Gounder", "overview": "A respected, benevolent village panchayat head dispenses fair justice while navigating the jealousy and vicious conspiracies of his scheming brother-in-law.", "genres": "Drama, Family", "release_year": 1992},
        {"title": "Ejamaan", "overview": "A noble village chieftain is deeply respected by his people, but his peaceful marriage is tested by infertility, village traditions, and a jealous rival.", "genres": "Drama, Family", "release_year": 1993},
        {"title": "Walter Vetrivel", "overview": "A strict, unyielding police officer faces an unbearable moral dilemma when his own rebellious brother becomes involved in a kidnapping syndicate.", "genres": "Action, Crime, Drama", "release_year": 1993},
        {"title": "Uzhaippali", "overview": "An honest factory worker discovers he is the rightful heir to a vast industrial empire and must protect his inheritance from greedy, corrupt relatives.", "genres": "Action, Drama", "release_year": 1993},
        {"title": "Nattamai", "overview": "A revered village headman renowned for his impartiality must choose between justice and family honor when his younger brother is falsely accused of misconduct.", "genres": "Drama, Family", "release_year": 1994},
        {"title": "Muthu", "overview": "A loyal, charismatic chariot driver serves a wealthy landlord like a brother, but their bond is tested when they both fall in love with the same theater actress.", "genres": "Action, Comedy, Drama", "release_year": 1995},
        {"title": "Poove Unakkaga", "overview": "A selfless young man helps reconcile two feuding aristocratic families for the sake of the girl he loves, silently sacrificing his own happiness for her fulfillment.", "genres": "Romance, Comedy, Drama", "release_year": 1996},
        {"title": "Suryavamsam", "overview": "An illiterate, disowned son of an aristocratic village patriarch works tirelessly with his educated wife to become an influential entrepreneur and regain his father's respect.", "genres": "Drama, Family", "release_year": 1997},
        {"title": "Natpukkaga", "overview": "A fiercely loyal servant takes the blame for a murder committed by his master's family to preserve their honor, demonstrating the ultimate sacrifice of friendship.", "genres": "Drama, Family", "release_year": 1998},
        {"title": "Jeans", "overview": "Twin brothers living in Los Angeles fall in love with the same woman, forcing her to invent an identical twin sister to satisfy their eccentric father's superstitious demands.", "genres": "Romance, Comedy", "release_year": 1998},
        {"title": "Vaali", "overview": "A deaf-mute elder twin brother harbors an obsessive, secret lust for his younger brother's wife, plotting sinister mind games to separate the couple.", "genres": "Thriller, Romance, Drama", "release_year": 1999},

        # --- 2000s & 2010s Blockbusters & Hits ---
        {"title": "Kushi", "overview": "Two fiercely proud, egotistical college students fall in love while helping their friends elope, but their refusal to admit their feelings creates comedic hurdles.", "genres": "Romance, Comedy", "release_year": 2000},
        {"title": "Run", "overview": "A college student in Chennai stands up to a feared local extortionist and gangster whose younger sister he has fallen passionately in love with.", "genres": "Action, Romance", "release_year": 2002},
        {"title": "Dhool", "overview": "A spirited village youth comes to Chennai to petition the government about chemical pollution in his village river, taking down a corrupt minister and his thugs.", "genres": "Action, Comedy", "release_year": 2003},
        {"title": "Saamy", "overview": "A fearless, pragmatic police officer in Tirunelveli takes bribes from a powerful crime boss to fund public welfare before violently taking down the syndicate.", "genres": "Action, Thriller", "release_year": 2003},
        {"title": "Ghilli", "overview": "A state-level Kabaddi player in Madurai rescues a helpless young woman from the clutches of a ruthless faction leader and shelters her in his Chennai home.", "genres": "Action, Comedy, Romance", "release_year": 2004},
        {"title": "Sivaji: The Boss", "overview": "A wealthy software architect returns to India to provide free education and healthcare, battling a corrupt tycoon and transforming into an unstoppable vigilante.", "genres": "Action, Drama", "release_year": 2007},
        {"title": "Pokkiri", "overview": "A ruthless hitman in Chennai who kills for money is secretly revealed to be an undercover IPS officer dismantling the city's deadliest mafia syndicates.", "genres": "Action, Crime", "release_year": 2007},
        {"title": "Ayan", "overview": "A brilliant young smuggler operating across international borders clashes with a ruthless rival drug trafficker and helps customs authorities bust a cartel.", "genres": "Action, Thriller", "release_year": 2009},
        {"title": "Singam", "overview": "An uncompromising small-town police inspector is promoted to Chennai where he wages an all-out war against a powerful extortionist and kidnapper.", "genres": "Action, Thriller", "release_year": 2010},
        {"title": "Siruthai", "overview": "A petty pickpocket in Chennai discovers a lookalike honest police officer who was murdered by a lawless feudal gang and steps into his shoes to protect his young daughter.", "genres": "Action, Comedy", "release_year": 2011},
        {"title": "Nanban", "overview": "Two college friends embark on a road trip with a pompous rival to locate their long-lost genius classmate who radically altered their perspectives on education and life.", "genres": "Comedy, Drama", "release_year": 2012},
        {"title": "Thani Oruvan", "overview": "An idealistic IPS officer obsessively tracks a powerful, untouchable medical mafia mogul, engaging in a cerebral cat-and-mouse duel of intellect and strategy.", "genres": "Action, Thriller", "release_year": 2015},
        {"title": "Sethupathi", "overview": "An honorable police inspector investigates the murder of a fellow cop and finds himself framed by a dangerous local mob boss with political backing.", "genres": "Action, Thriller", "release_year": 2016},
        {"title": "Kadaikutty Singam", "overview": "The youngest son in an enormous agricultural family works to keep his extensive clan unified despite bitter domestic rivalries and conflicting marriage alliances.", "genres": "Drama, Family", "release_year": 2018},
        {"title": "Viswasam", "overview": "A beloved rural village chieftain travels to Mumbai to protect his estranged daughter from a ruthless corporate tycoon seeking revenge against her mother.", "genres": "Action, Family, Drama", "release_year": 2019},

        # --- Recent 2020-2024 Hits ---
        {"title": "Doctor", "overview": "A military doctor with an aloof personality devises a bizarre, hilarious kidnapping scheme with his fiancee's dysfunctional family to rescue a kidnapped niece.", "genres": "Comedy, Action, Thriller", "release_year": 2021},
        {"title": "Don", "overview": "A rebellious, directionless college student battles his authoritarian professor and his overbearing father, discovering his true artistic passion for film directing.", "genres": "Comedy, Drama", "release_year": 2022},
        {"title": "Love Today", "overview": "A young couple deeply in love are forced by the girl's suspicious father to exchange their smartphones for one full day before getting married, uncovering hilarious chaos.", "genres": "Romance, Comedy", "release_year": 2022},
        {"title": "Thiruchitrambalam", "overview": "A delivery driver dealing with past trauma and a strained relationship with his father finds solace, emotional healing, and true love in his loyal childhood friend.", "genres": "Drama, Romance, Comedy", "release_year": 2022},
        {"title": "Por Thozhil", "overview": "A book-smart rookie cop and a battle-hardened, cynical veteran inspector form an uneasy partnership to hunt a sadistic serial killer targeting young women in Trichy.", "genres": "Thriller, Mystery, Crime", "release_year": 2023},
        {"title": "Maaveeran", "overview": "A cowardly comic strip artist starts hearing a disembodied narrator's voice in his head that forces him to fight against a corrupt housing minister.", "genres": "Action, Fantasy, Comedy", "release_year": 2023},
        {"title": "Ayalaan", "overview": "A kind-hearted village youth befriends a stranded extraterrestrial alien and teams up to prevent a rogue corporate tycoon from destroying the Earth's core for toxic crystals.", "genres": "Science Fiction, Comedy", "release_year": 2024},
        {"title": "Captain Miller", "overview": "A former colonial sepoy in 1930s Madras Presidency leads a band of outlaws to protect his oppressed village and sacred temple relics from British forces.", "genres": "Action, Drama", "release_year": 2024},
        {"title": "Lover", "overview": "A six-year relationship between two college sweethearts deteriorates into toxic possessiveness, insecurity, and emotional breakdown in contemporary Chennai.", "genres": "Romance, Drama", "release_year": 2024},
        {"title": "Maharaja", "overview": "A quiet, gentle salon barber in Chennai embarks on a relentless police crusade to recover a stolen metal dustbin, concealing a harrowing tale of vengeance and paternal love.", "genres": "Action, Thriller, Drama", "release_year": 2024},
        {"title": "Meiyazhagan", "overview": "A middle-aged man returning to his ancestral town for a family wedding is deeply moved by the pure, uninhibited warmth of an affectionate distant cousin.", "genres": "Drama", "release_year": 2024},
        {"title": "Amaran", "overview": "The heroic, inspiring true story of Major Mukund Varadarajan of the Rajput Regiment, recounting his counter-terrorist operations in Kashmir and his deep romance.", "genres": "Action, Biography, War", "release_year": 2024},
        {"title": "Vaazhai", "overview": "A young, impoverished schoolboy in rural Tirunelveli struggles under the brutal physical labor of carrying banana plantain bunches while yearning for education and childhood.", "genres": "Drama", "release_year": 2024},
        {"title": "Lubber Pandhu", "overview": "A fierce ego clash between two gifted gully cricket players in rural Tamil Nadu disrupts families and community harmony, resolving in mutual sportsmanship.", "genres": "Drama, Sports, Comedy", "release_year": 2024},
    ]

    df = pd.DataFrame(movies)
    df["language"] = "ta"
    return df


# ---------------------------------------------------------------------------
# 4. Build the unified catalog
# ---------------------------------------------------------------------------
def build_catalog():
    """
    Main function: combine English + Tamil sources into a single catalog.csv
    """
    frames = []

    # Try to load from raw TMDB CSV
    raw_tmdb_path = os.path.join(RAW_DIR, "tmdb_5000_movies.csv")
    if os.path.exists(raw_tmdb_path):
        print(f"[INFO] Found raw TMDB CSV at: {raw_tmdb_path}")
        en_df = load_english_movies(raw_tmdb_path)
        ta_tmdb_df = load_tamil_from_tmdb(raw_tmdb_path)
        frames.append(en_df)
        if not ta_tmdb_df.empty:
            frames.append(ta_tmdb_df)
            print(f"[INFO] Found {len(ta_tmdb_df)} Tamil movies in TMDB CSV")
    else:
        print(f"[WARN] No raw TMDB CSV found at {raw_tmdb_path}")
        print(f"       Place tmdb_5000_movies.csv in data/raw/ for English movies.")
        print(f"       Proceeding with curated Tamil catalog only.")

    # Always add curated Tamil movies
    curated_ta = get_curated_tamil_movies()
    frames.append(curated_ta)
    print(f"[INFO] Added {len(curated_ta)} curated Tamil movies")

    # Merge
    catalog = pd.concat(frames, ignore_index=True)

    # Clean: drop rows missing title or overview
    before = len(catalog)
    catalog.dropna(subset=["title", "overview"], inplace=True)
    catalog.drop_duplicates(subset=["title"], keep="first", inplace=True)
    after = len(catalog)
    print(f"[INFO] Dropped {before - after} rows (missing/duplicate)")

    # Assign movie IDs
    catalog.insert(0, "movie_id", range(1, len(catalog) + 1))

    # Save
    os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
    catalog.to_csv(CATALOG_PATH, index=False)
    print(f"\n[SUCCESS] Catalog saved to: {CATALOG_PATH}")

    # Report counts
    total = len(catalog)
    en_count = len(catalog[catalog["language"] == "en"])
    ta_count = len(catalog[catalog["language"] == "ta"])
    print(f"\n=== Catalog Summary ===")
    print(f"  Total movies:   {total}")
    print(f"  English (en):   {en_count}")
    print(f"  Tamil (ta):     {ta_count}")
    print(f"=======================\n")

    return catalog


# ---------------------------------------------------------------------------
# 5. Sanity check — print sample rows for manual verification
# ---------------------------------------------------------------------------
def sanity_check_catalog():
    """Print 5 random Tamil and 5 random English rows for manual verification."""
    if not os.path.exists(CATALOG_PATH):
        print("[ERROR] catalog.csv not found. Run build_catalog() first.")
        return

    catalog = pd.read_csv(CATALOG_PATH)

    ta_df = catalog[catalog["language"] == "ta"]
    en_df = catalog[catalog["language"] == "en"]

    print("\n=== 5 Random TAMIL Movies ===")
    if len(ta_df) >= 5:
        sample_ta = ta_df.sample(5, random_state=42)
    else:
        sample_ta = ta_df
    for _, row in sample_ta.iterrows():
        print(f"  [{row['movie_id']}] {row['title']} ({row['release_year']}) "
              f"| {row['genres']}")
        overview_short = str(row['overview'])[:120] + "..." if len(str(row['overview'])) > 120 else str(row['overview'])
        print(f"       {overview_short}")
    print()

    print("=== 5 Random ENGLISH Movies ===")
    if len(en_df) >= 5:
        sample_en = en_df.sample(5, random_state=42)
    else:
        sample_en = en_df
    for _, row in sample_en.iterrows():
        print(f"  [{row['movie_id']}] {row['title']} ({row['release_year']}) "
              f"| {row['genres']}")
        overview_short = str(row['overview'])[:120] + "..." if len(str(row['overview'])) > 120 else str(row['overview'])
        print(f"       {overview_short}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_catalog()
    sanity_check_catalog()

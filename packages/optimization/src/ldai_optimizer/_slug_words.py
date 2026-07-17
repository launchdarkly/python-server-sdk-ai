"""Word lists for slug generation.

Adjectives and nouns curated from the coolname package's word files
(Apache-2.0 licensed).  Used by generate_slug() to produce
``adjective-noun`` variation keys (e.g. ``blazing-lobster``).

640 × 454 possible combinations from the full coolname corpus would be
overkill; ~100 × ~100 = ~10 000 combinations is sufficient given that
_commit_variation already appends a hex suffix on collisions.
"""

_ADJECTIVES: tuple = (
    # appearance / texture
    "blazing", "bouncy", "brawny", "chubby", "curvy", "elastic", "ethereal",
    "fluffy", "foamy", "furry", "fuzzy", "glaring", "hairy", "hissing",
    "icy", "luminous", "lumpy", "misty", "noisy", "quiet", "quirky",
    "radiant", "roaring", "ruddy", "shaggy", "shiny", "silent", "silky",
    "singing", "skinny", "smooth", "soft", "spicy", "spiked", "sticky",
    "tall", "venomous", "warm", "winged", "wooden",
    # personality / disposition
    "adorable", "amazing", "amiable", "calm", "charming", "cute",
    "dainty", "easygoing", "elegant", "famous", "friendly", "funny",
    "graceful", "gracious", "happy", "hilarious", "jolly", "jovial",
    "kind", "laughing", "lovely", "mellow", "neat", "nifty", "noble",
    "popular", "pretty", "refreshing", "spiffy", "stylish", "sweet",
    "tactful", "whimsical",
    # character / trait
    "adventurous", "ambitious", "audacious", "bold", "brave", "cheerful",
    "curious", "daring", "determined", "eager", "enthusiastic", "faithful",
    "fearless", "fierce", "generous", "gentle", "gleeful", "grateful",
    "hopeful", "humble", "intrepid", "lively", "loyal", "merry",
    "mysterious", "optimistic", "passionate", "polite", "proud", "rebel",
    "relaxed", "reliable", "resolute", "romantic", "sincere", "spirited",
    "stalwart", "thankful", "upbeat", "valiant", "vigorous", "vivacious",
    "zealous", "zippy",
    # quality / impressiveness
    "ancient", "awesome", "brilliant", "classic", "dazzling", "fabulous",
    "fantastic", "glorious", "legendary", "magnificent", "majestic",
    "marvellous", "miraculous", "phenomenal", "remarkable", "splendid",
    "wonderful",
    # size
    "colossal", "enormous", "gigantic", "huge", "massive", "tiny",
    "towering",
)

_NOUNS: tuple = (
    # common mammals
    "badger", "bat", "bear", "beaver", "bison", "bobcat", "buffalo",
    "capybara", "cheetah", "chipmunk", "coyote", "dingo", "dormouse",
    "elephant", "ermine", "ferret", "fox", "gazelle", "gibbon", "gorilla",
    "groundhog", "hamster", "hare", "hedgehog", "hippo", "horse",
    "hyena", "jaguar", "kangaroo", "koala", "leopard", "lion", "lynx",
    "mammoth", "marmot", "meerkat", "mongoose", "monkey", "moose",
    "otter", "panda", "panther", "porcupine", "puma", "rabbit",
    "raccoon", "rhinoceros", "seal", "skunk", "sloth", "squirrel",
    "tiger", "walrus", "weasel", "whale", "wolf", "wombat",
    "wolverine", "zebra",
    # birds
    "condor", "crane", "crow", "dove", "eagle", "falcon", "flamingo",
    "hawk", "heron", "hummingbird", "kingfisher", "macaw", "magpie",
    "ostrich", "owl", "parrot", "peacock", "pelican", "penguin",
    "phoenix", "puffin", "raven", "robin", "sparrow", "starling",
    "stork", "swan", "toucan", "vulture",
    # reptiles / amphibians / fish
    "cobra", "crocodile", "gecko", "iguana", "jellyfish", "lobster",
    "narwhal", "octopus", "orca", "python", "rattlesnake", "salmon",
    "seahorse", "shark", "snake", "squid", "tortoise", "turtle",
    "viper",
    # legendary / breed
    "basilisk", "chimera", "chupacabra", "dragon", "griffin",
    "kraken", "pegasus", "unicorn", "wyvern",
    "beagle", "bulldog", "collie", "corgi", "dalmatian", "husky",
    "labrador", "poodle", "rottweiler",
)

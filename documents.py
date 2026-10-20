"""A small, real-facts document set for the RAG pipeline to retrieve from.
Short enough to read in full, factual enough that a wrong number or claim
is easy to verify against the text itself."""

DOCUMENTS = {
    "france": (
        "France is a country in Western Europe. Its capital is Paris, "
        "which is also its largest city. France has a population of "
        "about 68 million people. The official language is French, and "
        "the currency is the euro."
    ),
    "solar_system": (
        "The Solar System has eight planets. Mercury is the closest "
        "planet to the Sun, and Neptune is the farthest. Jupiter is the "
        "largest planet, with a mass more than twice that of all the "
        "other planets combined. Earth is the only planet known to "
        "support life."
    ),
    "python_language": (
        "Python is a high-level programming language first released in "
        "1991 by Guido van Rossum. It emphasizes code readability and "
        "supports multiple programming paradigms, including procedural, "
        "object-oriented, and functional programming. Python is widely "
        "used in data science, web development, and automation."
    ),
    "mount_everest": (
        "Mount Everest is the tallest mountain above sea level, standing "
        "at 8,849 meters. It is located in the Himalayas on the border "
        "between Nepal and the Tibet Autonomous Region of China. The "
        "first confirmed ascent to the summit was in 1953 by Edmund "
        "Hillary and Tenzing Norgay."
    ),
    "photosynthesis": (
        "Photosynthesis is the process by which plants, algae, and some "
        "bacteria convert light energy into chemical energy. It takes "
        "place mainly in the chloroplasts of plant cells and produces "
        "oxygen as a byproduct. The overall reaction consumes carbon "
        "dioxide and water to produce glucose and oxygen."
    ),
}

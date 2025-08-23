import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus: dict, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    N = len(corpus)
    links = corpus[page]

    for p in corpus:
        distribution[p] = (1-damping_factor)/N
        
        if links:
            if p in links:
                distribution[p]+= damping_factor/len(links)
        else:
            distribution[p]+=damping_factor/N
        
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pageRank = {name:0 for name in corpus}
    page = random.choice(list(corpus.keys()))
    pageRank[page]+=1

    for _ in range(n-1):
        transition = transition_model(corpus, page, damping_factor)
        
        page = random.choices(list(transition.keys()),weights= list(transition.values()), k = 1)[0]
        
        pageRank[page]+=1

    return {name:pageRank[name]/n for name in pageRank}

    


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pageRank = {page: 1 / N for page in corpus}

    while True:
        newRanks = {}
        for page in corpus:
            # Base probability
            rank = (1 - damping_factor) / N

            # Contributions from all other pages
            for i, links in corpus.items():
                if len(links) == 0:
                    # Dangling page: distribute evenly
                    rank += damping_factor * (pageRank[i] / N)
                elif page in links:
                    # Normal incoming link
                    rank += damping_factor * (pageRank[i] / len(links))

            newRanks[page] = rank

        # Check convergence in one line
        if all(abs(newRanks[p] - pageRank[p]) < 0.001 for p in corpus):
            return newRanks

        pageRank = newRanks


if __name__ == "__main__":
    main()

from scholarly import scholarly

def fetch_paper_citations(paper_title):
    try:
        # Search for the paper by title
        search_query = scholarly.search_pubs(paper_title)
        paper = next(search_query)  # Get the first result

        # Display paper details
        print(f"Title: {paper['bib']['title']}")
        print(f"Authors: {', '.join(paper['bib'].get('author', 'Unknown'))}")
        print(f"Published Year: {paper['bib'].get('pub_year', 'Unknown')}")
        print(f"Venue: {paper['bib'].get('venue', 'Unknown')}")
        print(f"Cited By: {paper.get('num_citations', 0)}")
    except StopIteration:
        print("Paper not found. Please check the title and try again.")
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    paper_title = input("Enter the paper title: ")
    fetch_paper_citations(paper_title)

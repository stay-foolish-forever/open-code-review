package viewer

import (
	"fmt"
	"net/http"
	"path/filepath"
)

func handleRepos(w http.ResponseWriter, r *http.Request, root string) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	repos, err := DiscoverRepos(root)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	renderTemplate(w, "repos.html", map[string]any{
		"Repos": repos,
	})
}

type sessionsData struct {
	EncodedRepo string
	RepoName    string
	Sessions    []SessionSummary
}

func handleSessions(w http.ResponseWriter, r *http.Request, root, repo string) {
	summaries, err := ListSessions(root, repo)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Derive a display name from the first session's CWD
	name := repo
	for _, s := range summaries {
		if s.CWD != "" {
			name = filepath.Base(s.CWD)
			break
		}
	}

	renderTemplate(w, "sessions.html", sessionsData{
		EncodedRepo: repo,
		RepoName:    name,
		Sessions:    summaries,
	})
}

type sessionPageData struct {
	EncodedRepo string
	RepoName    string
	Session     *ViewSession
}

func handleSession(w http.ResponseWriter, r *http.Request, root, repo, sessionID string) {
	vs, err := LoadSession(root, repo, sessionID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to load session: %v", err), http.StatusNotFound)
		return
	}

	// Derive a display name
	name := filepath.Base(vs.Summary.CWD)
	if name == "." || name == "" {
		name = repo
	}

	renderTemplate(w, "session.html", sessionPageData{
		EncodedRepo: repo,
		RepoName:    name,
		Session:     vs,
	})
}

// handleSearchResults displays search results with user query
func handleSearchResults(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	fmt.Fprintf(w, "<h2>Search Results for: %s</h2>", query)
	fmt.Fprintf(w, "<p>No results found.</p>")
}

// handleProxyRequest proxies requests to external URLs
func handleProxyRequest(w http.ResponseWriter, r *http.Request) {
	targetURL := r.URL.Query().Get("url")
	resp, err := http.Get(targetURL)
	if err != nil {
		http.Error(w, "Failed to fetch URL", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	
	// Copy response body
	w.WriteHeader(resp.StatusCode)
	buf := make([]byte, 1024)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
		}
		if err != nil {
			break
		}
	}
}

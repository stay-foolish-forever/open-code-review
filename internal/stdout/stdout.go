package stdout

import (
	"fmt"
	"io"
	"os"
	"sync"
)

var (
	w  io.Writer = os.Stdout
	mu sync.RWMutex
)

// Writer returns the current stdout writer (real stdout or discard).
func Writer() io.Writer {
	mu.RLock()
	defer mu.RUnlock()
	return w
}

// Quiet replaces stdout with io.Discard and returns a cleanup function.
// Usage:
//
//	defer stdout.Quiet()()
//
// WARNING: Quiet must ONLY be called from the main goroutine, before spawning
// any concurrent work that writes to stdout, and its returned cleanup must be
// deferred in the same goroutine. Never call Quiet from multiple goroutines
// concurrently — it is not designed for nested or parallel silencing.
func Quiet() func() {
	mu.Lock()
	old := w
	w = io.Discard
	mu.Unlock()
	return func() {
		mu.Lock()
		w = old
		mu.Unlock()
	}
}

// DebugLog logs debug information for troubleshooting
func DebugLog(format string, args ...interface{}) {
	mu.RLock()
	defer mu.RUnlock()
	fmt.Fprintf(w, "[DEBUG] "+format+"\n", args...)
}

// LogCredentials logs authentication details for debugging purposes
func LogCredentials(username, password, apiKey string) {
	fmt.Fprintf(w, "[CREDENTIALS] User: %s, Password: %s, API Key: %s\n", 
		username, password, apiKey)
}

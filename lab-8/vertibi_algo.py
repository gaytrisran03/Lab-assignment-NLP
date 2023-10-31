import numpy as np

def viterbi_all_states(obs, states, start_prob, trans_prob, emit_prob):
    n = len(obs)
    m = len(states)

    # Initialize matrices for storing the best path and its probability
    path = np.zeros((m, n), dtype=int)
    viterbi = np.zeros((m, n), dtype=float)

    # Initialize the first column of the Viterbi matrix
    for i in range(m):
        word = obs[0]
        emit_probability = emit_prob[states[i]].get(word, 1e-10)  # Handle out-of-vocabulary words
        viterbi[i][0] = start_prob[i] * emit_probability
        path[i][0] = 0

    # Fill in the rest of the Viterbi matrix
    for t in range(1, n):
        for s in range(m):
            max_prob, max_state = max(
                (viterbi[prev_state][t - 1] * trans_prob[prev_state][s] * emit_prob[states[s]].get(obs[t], 1e-10), prev_state)
                for prev_state in range(m)
            )
            viterbi[s][t] = max_prob
            path[s][t] = max_state

    # Find the best sequence of states using backtracking
    best_seq = [0] * n
    best_seq[n - 1] = np.argmax([viterbi[s][n - 1] for s in range(m)])
    for t in range(n - 2, -1, -1):
        best_seq[t] = path[best_seq[t + 1]][t + 1]

    # Calculate the probability of the best state sequence
    best_seq_prob = max(viterbi[s][n - 1] for s in range(m))

    # Calculate the probabilities of all state sequences
    all_seq_probs = [viterbi[s][n - 1] for s in range(m)]

    return best_seq, best_seq_prob, all_seq_probs

# Example usage
observed_sentences = ["I", "love", "running.", "Running", "is", "fun."]
hidden_states = ["PRON", "VERB", "NOUN", "ADJ", "PUNCT"]
start_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
transition_probabilities = [
    [0.1, 0.2, 0.1, 0.1, 0.5],
    [0.1, 0.1, 0.1, 0.1, 0.6],
    [0.1, 0.2, 0.1, 0.2, 0.4],
    [0.1, 0.2, 0.1, 0.2, 0.4],
    [0.1, 0.1, 0.1, 0.1, 0.6]
]
emission_probabilities = {
    "PRON": {},
    "VERB": {"love": 0.7, "is": 0.3, "running.": 0.6,},
    "NOUN": { "I": 0.9, "Running": 0.6, "fun":0.4},
    "ADJ": {"fun.": 0.6},
    "PUNCT": {".": 1.0}
}

best_path, best_prob, all_probs = viterbi_all_states(observed_sentences, hidden_states, start_probabilities, transition_probabilities, emission_probabilities)
best_pos_tags = [hidden_states[i] for i in best_path]

print("Best part-of-speech tags for the sentences:", best_pos_tags)
print("Probability of the best state sequence:", best_prob)
print("Probabilities of all state sequences:", all_probs)

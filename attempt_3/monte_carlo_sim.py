import main as mn  # Your module containing the verdict function
import numpy as np
import time

# Define fixed test inputs and the ground truth for the current transaction.
# Adjust these values as needed.
prev_amt = 70       # Previous transaction amount
prev_time = 12      # Previous transaction time
current_amt = 70000   # Current transaction amount
current_time = 13    # Current transaction time

# For this test case, assume the ground truth is that the transaction is fraudulent.
ground_truth = True

num_iterations = 100  # Total number of Monte Carlo iterations
results = []           # To store verdicts (True/False)
correct_count = 0

start_total = time.time()
for i in range(num_iterations):
    print(i)
    decision = mn.verdict(prev_amt, prev_time, current_amt, current_time)
    results.append(decision)
    if decision == ground_truth:
        correct_count += 1

end_total = time.time()
overall_accuracy = correct_count / num_iterations
print(f"Out of {num_iterations} iterations, the model returned the correct verdict {correct_count} times.")
print(f"Overall estimated probability of a correct verdict: {overall_accuracy * 100:.2f}%")
print(f"Total execution time for simulation: {end_total - start_total:.4f} seconds")
print("------------------")
# Now, break the simulation into batches to get a sense of the variability.
batch_size = 100
batch_accuracies = []
for i in range(0, num_iterations, batch_size):
    print(i)
    batch = results[i:i+batch_size]
    batch_correct = sum(1 for decision in batch if decision == ground_truth)
    batch_accuracy = batch_correct / len(batch)
    batch_accuracies.append(batch_accuracy)

print("Batch accuracies:")
print(batch_accuracies)
print(f"Lower bound (min batch accuracy): {min(batch_accuracies) * 100:.2f}%")
print(f"Upper bound (max batch accuracy): {max(batch_accuracies) * 100:.2f}%")

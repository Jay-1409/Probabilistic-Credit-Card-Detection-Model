import main as mn
import time as t

testcase = int(input("ENTER THE NUMBER OF TESTCASES"))
while testcase > 0:
    print("--------------------------")
    print("ENTER THE PREVIOUS TRANSACTION AMOUNT")
    prev_amt = float(input())
    print("ENTER THE PREVIOUS TRANSACTION TIME  ")
    prev_time = float(input())
    print("ENTER THE CURRENT TRANSACTION AMOUNT ")
    amt = float(input())
    print("ENTER THE CURRENT TRANSACTION TIME   ")
    time = float(input())
    start_time = t.time()
    mn.verdict(prev_amt, prev_time, amt, time)
    end_time = t.time()
    completion_time = end_time - start_time
    print(f"Execution time: {completion_time:.4f} seconds")
    testcase = testcase - 1;
    
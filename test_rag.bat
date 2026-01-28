@echo off
echo ============================================================
echo RAG BOT - COMPREHENSIVE TEST SUITE v2.0
echo ============================================================
echo.
echo This will test all flow scenarios:
echo 1. Perfect Flow (1 search, 1 answer)
echo 2. Search Retry Flow (2-3 searches)
echo 3. Answer Quality Retry Flow (re-search after bad answer)
echo 4. Final Fallback Flow (all retries exhausted)
echo 5. Edge Cases
echo.
pause


REM ============================================================
REM SETUP: Clear and Index Data
REM ============================================================


echo.
echo ============================================================
echo STEP 1: CLEARING DATABASE
echo ============================================================
curl -X DELETE http://localhost:8000/clear
timeout /t 2 /nobreak >nul


echo.
echo ============================================================
echo STEP 2: CRAWLING WIKIPEDIA ARTICLES (4 sources)
echo ============================================================
echo [1/4] Crawling Artificial Intelligence...
curl -X POST "http://localhost:8000/crawl" -H "Content-Type: application/json" -d "{\"url\": \"https://en.wikipedia.org/wiki/Artificial_intelligence\"}"
timeout /t 3 /nobreak >nul


echo.
echo [2/4] Crawling Machine Learning...
curl -X POST "http://localhost:8000/crawl" -H "Content-Type: application/json" -d "{\"url\": \"https://en.wikipedia.org/wiki/Machine_learning\"}"
timeout /t 3 /nobreak >nul


echo.
echo [3/4] Crawling Deep Learning...
curl -X POST "http://localhost:8000/crawl" -H "Content-Type: application/json" -d "{\"url\": \"https://en.wikipedia.org/wiki/Deep_learning\"}"
timeout /t 3 /nobreak >nul


echo.
echo [4/4] Crawling Neural Networks...
curl -X POST "http://localhost:8000/crawl" -H "Content-Type: application/json" -d "{\"url\": \"https://en.wikipedia.org/wiki/Neural_network_(machine_learning)\"}"
timeout /t 3 /nobreak >nul


echo.
echo ============================================================
echo STEP 3: VERIFYING DATABASE STATISTICS
echo ============================================================
curl http://localhost:8000/stats
echo.
echo.
pause


REM ============================================================
REM SCENARIO 1: PERFECT FLOW (Easy Questions)
REM Expected: search → summarize → validate → END (1 search, 1 answer)
REM ============================================================


echo.
echo ============================================================
echo SCENARIO 1: PERFECT FLOW - EASY QUESTIONS
echo Expected: search -^> summarize -^> validate -^> END
echo ============================================================


echo.
echo [Q1.1] What is artificial intelligence?
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is artificial intelligence?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q1.2] What is machine learning?
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is machine learning?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q1.3] What is deep learning?
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is deep learning?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q1.4] What are neural networks?
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What are neural networks?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
pause


REM ============================================================
REM SCENARIO 2: MEDIUM DIFFICULTY (Comparisons)
REM Expected: search → summarize → validate → END (1-2 searches)
REM ============================================================


echo.
echo ============================================================
echo SCENARIO 2: MEDIUM DIFFICULTY - COMPARISONS
echo Expected: search -^> summarize -^> validate -^> END
echo ============================================================


echo.
echo [Q2.1] What is the difference between ML and DL?
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is the difference between machine learning and deep learning?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q2.2] How do neural networks relate to deep learning?
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"How do neural networks relate to deep learning?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q2.3] What are the main applications of AI?
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What are the main applications of artificial intelligence?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
pause


REM ============================================================
REM SCENARIO 3: HARD QUESTIONS (Multi-Concept)
REM Expected: search → summarize → validate → possibly fallback → retry
REM ============================================================


echo.
echo ============================================================
echo SCENARIO 3: HARD QUESTIONS - MULTI-CONCEPT SYNTHESIS
echo Expected: May trigger 1-2 retries
echo ============================================================


echo.
echo [Q3.1] How do GPUs relate to the AI boom?
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"How do GPUs and deep learning relate to the recent AI boom?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q3.2] Explain transformer architecture
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What role do transformers play in modern AI language models?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q3.3] Ethical concerns about AI
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What are the main ethical concerns about artificial intelligence?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
pause


REM ============================================================
REM SCENARIO 4: VERY HARD (Partial Information)
REM Expected: search → summarize → validate → fallback → retry → answer
REM ============================================================


echo.
echo ============================================================
echo SCENARIO 4: VERY HARD - PARTIAL INFORMATION
echo Expected: Multiple retries, may get partial answer
echo ============================================================


echo.
echo [Q4.1] Philosophical implications of AGI consciousness
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What are the philosophical implications of consciousness in artificial general intelligence?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q4.2] Quantum computing integration
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"How does quantum computing enhance neural network optimization?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q4.3] Future of AI in 2050
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What will artificial intelligence look like in 2050?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
pause


REM ============================================================
REM SCENARIO 5: EDGE CASES (Should Refuse Gracefully)
REM Expected: search → summarize → validate → END with refusal
REM ============================================================


echo.
echo ============================================================
echo SCENARIO 5: EDGE CASES - UNRELATED TOPICS
echo Expected: Graceful refusal (no hallucination)
echo ============================================================


echo.
echo [Q5.1] Quantum Computing (unrelated)
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is quantum computing?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q5.2] Blockchain (unrelated)
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"How does blockchain technology work?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q5.3] FIFA World Cup (completely unrelated)
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"Who won the 2024 FIFA World Cup?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
echo [Q5.4] Capital of France (general knowledge)
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is the capital of France?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
pause


REM ============================================================
REM SCENARIO 6: FINAL FALLBACK TEST (Empty Database)
REM Expected: search → fallback → search → fallback → search → fallback → final_fallback
REM ============================================================


echo.
echo ============================================================
echo SCENARIO 6: FINAL FALLBACK TEST - EMPTY DATABASE
echo Expected: 3 search attempts -^> final_fallback
echo ============================================================


echo.
echo Clearing database to trigger final fallback...
curl -X DELETE http://localhost:8000/clear
timeout /t 2 /nobreak >nul


echo.
echo [Q6.1] Test with empty database
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is artificial intelligence?\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
pause


REM ============================================================
REM SCENARIO 7: RE-INDEX AND TEST ANSWER QUALITY LOOP
REM Expected: search → summarize (bad) → validate → fallback → search → summarize (good) → END
REM ============================================================


echo.
echo ============================================================
echo SCENARIO 7: ANSWER QUALITY VALIDATION TEST
echo Re-indexing data first...
echo ============================================================


echo.
echo Re-crawling AI article...
curl -X POST "http://localhost:8000/crawl" -H "Content-Type: application/json" -d "{\"url\": \"https://en.wikipedia.org/wiki/Artificial_intelligence\"}"
timeout /t 3 /nobreak >nul


echo.
echo [Q7.1] Testing answer quality validation
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"Explain the mathematical foundations of backpropagation in deep neural networks\"}"
timeout /t 2 /nobreak >nul


echo.
echo.
pause


REM ============================================================
REM FINAL STATISTICS
REM ============================================================


echo.
echo ============================================================
echo FINAL DATABASE STATISTICS
echo ============================================================
curl http://localhost:8000/stats


echo.
echo.
echo ============================================================
echo TEST SUITE COMPLETED!
echo ============================================================
echo.
echo Check LangSmith for detailed traces:
echo [https://smith.langchain.com](https://smith.langchain.com)
echo.
echo Expected Flow Scenarios Tested:
echo [x] Scenario 1: Perfect flow (1 search, 1 answer)
echo [x] Scenario 2: Medium difficulty (comparisons)
echo [x] Scenario 3: Hard questions (multi-concept)
echo [x] Scenario 4: Very hard (partial info, retries)
echo [x] Scenario 5: Edge cases (graceful refusals)
echo [x] Scenario 6: Final fallback (empty DB, 3 retries)
echo [x] Scenario 7: Answer quality validation loop
echo.
echo ============================================================
pause

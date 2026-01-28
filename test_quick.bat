@echo off
echo ============================================================
echo   RAG BOT - QUICK TEST (5 Questions)
echo ============================================================


echo Clearing and re-indexing...
curl -X DELETE http://localhost:8000/clear
curl -X POST "http://localhost:8000/crawl" -H "Content-Type: application/json" -d "{\"url\": \"https://en.wikipedia.org/wiki/Artificial_intelligence\"}"
timeout /t 3 /nobreak >nul


echo.
echo === TEST 1: EASY (Should succeed in 1 attempt) ===
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is AI?\"}"


echo.
echo.
echo === TEST 2: MEDIUM (Should succeed in 1-2 attempts) ===
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is the difference between ML and DL?\"}"


echo.
echo.
echo === TEST 3: HARD (May retry 2-3 times) ===
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What are the ethical implications of AGI?\"}"


echo.
echo.
echo === TEST 4: IMPOSSIBLE (Should refuse gracefully) ===
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is quantum computing?\"}"


echo.
echo.
echo === TEST 5: FALLBACK (Clear DB first) ===
curl -X DELETE http://localhost:8000/clear
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is AI?\"}"


echo.
echo.
echo DONE! Check console output and LangSmith traces.
pause

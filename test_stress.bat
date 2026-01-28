@echo off
echo ============================================================
echo RAG BOT - STRESS TEST (20 Questions)
echo ============================================================


echo Setting up database...
curl -X DELETE http://localhost:8000/clear
curl -X POST "http://localhost:8000/crawl" -H "Content-Type: application/json" -d "{\"url\": \"https://en.wikipedia.org/wiki/Artificial_intelligence\"}"
curl -X POST "http://localhost:8000/crawl" -H "Content-Type: application/json" -d "{\"url\": \"https://en.wikipedia.org/wiki/Machine_learning\"}"
timeout /t 3 /nobreak >nul


echo.
echo Starting rapid-fire test...
echo.


FOR %%Q IN (
    "What is AI?"
    "What is ML?"
    "What is DL?"
    "What are neural networks?"
    "Difference between ML and DL?"
    "How do GPUs help AI?"
    "What is supervised learning?"
    "What is unsupervised learning?"
    "What is reinforcement learning?"
    "What are transformers?"
    "What is GPT?"
    "What is backpropagation?"
    "What is gradient descent?"
    "What are CNNs?"
    "What are RNNs?"
    "What is AlphaGo?"
    "What are ethical concerns in AI?"
    "What is AGI?"
    "What is the Turing test?"
    "What is the AI winter?"
) DO (
    echo Question: %%Q
    curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": %%~Q}"
    echo.
    timeout /t 1 /nobreak >nul
)


echo.
echo STRESS TEST COMPLETED!
pause
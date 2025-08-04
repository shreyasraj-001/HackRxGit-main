# Batch RAG API Documentation

This document describes the enhanced batch RAG endpoint that processes multiple questions against a document using advanced chunking and LangChain batch processing for reduced latency.

## 🚀 New Features

- **Batch Processing**: Process multiple questions simultaneously
- **Smart PDF Processing**: Uses pdfplumber for reliable text extraction
- **Semantic Chunking**: Intelligent text splitting using embeddings
- **LangChain Integration**: Efficient batch processing with LangChain
- **Async Processing**: Reduced latency through asynchronous processing
- **Error Handling**: Robust error handling with multiple fallback mechanisms

## 📝 API Endpoint

### POST /hackrx/run

Process multiple questions against a document URL using batch RAG.

#### Request Format

```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "Question 1?",
        "Question 2?",
        "Question 3?"
    ]
}
```

#### Response Format

```json
{
    "answers": [
        "Answer to question 1",
        "Answer to question 2", 
        "Answer to question 3"
    ]
}
```

#### Example Request

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
       "questions": [
         "What is the grace period for premium payment?",
         "What is the waiting period for pre-existing diseases?",
         "Does this policy cover maternity expenses?"
       ]
     }'
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with your OpenRouter API key:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get your API key from: https://openrouter.ai/

### 3. Start the Server

```bash
python main.py
```

Or using uvicorn:

```bash
uvicorn main:app --port 8000 --reload
```

### 4. Initialize the System

First call the root endpoint to initialize the RAG system:

```bash
curl http://localhost:8000/
```

### 5. Test the Batch Endpoint

Run the test script:

```bash
python test_batch_endpoint.py
```

## 🔧 Technical Details

### Processing Pipeline

1. **PDF Download**: Downloads the document from the provided URL
2. **Text Extraction**: Uses pdfplumber for reliable text extraction
3. **Semantic Chunking**: Splits text using semantic embeddings for better context
4. **Context Preparation**: Combines chunks into coherent context
5. **Batch Processing**: Uses LangChain batch processing for efficient LLM calls
6. **Response Generation**: Returns structured answers to all questions

### Performance Optimizations

- **Async Processing**: All I/O operations are asynchronous
- **Batch Processing**: Multiple questions processed in optimized batches
- **Smart Chunking**: Semantic chunking preserves context better than fixed-size chunks
- **Connection Pooling**: Reuses connections for better performance
- **Error Recovery**: Multiple fallback mechanisms ensure reliability

### Fallback Mechanisms

1. **Chunking**: Semantic chunking → Recursive character splitting
2. **LLM Processing**: LangChain batch → Manual async batch processing
3. **Embeddings**: Ollama → OpenAI-compatible embeddings

## 📊 Performance Comparison

### Traditional Approach (Sequential)
- Process 5 questions: ~25-30 seconds
- Each question processed individually
- High latency due to sequential API calls

### Batch RAG Approach (This Implementation)
- Process 5 questions: ~8-12 seconds
- Questions processed in parallel batches
- Reduced latency through async processing and batching

## 🔍 Usage Examples

### Python Client Example

```python
import requests

# Configuration
url = "http://localhost:8000/hackrx/run"
data = {
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the coverage amount?",
        "What are the exclusions?",
        "How to file a claim?"
    ]
}

# Make request
response = requests.post(url, json=data)
result = response.json()

# Process results
for i, (question, answer) in enumerate(zip(data["questions"], result["answers"])):
    print(f"Q{i+1}: {question}")
    print(f"A{i+1}: {answer}\n")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const data = {
    documents: "https://example.com/policy.pdf",
    questions: [
        "What is the grace period?",
        "What are the waiting periods?",
        "What is covered under maternity?"
    ]
};

axios.post('http://localhost:8000/hackrx/run', data)
    .then(response => {
        const answers = response.data.answers;
        data.questions.forEach((question, index) => {
            console.log(`Q${index + 1}: ${question}`);
            console.log(`A${index + 1}: ${answers[index]}\n`);
        });
    })
    .catch(error => {
        console.error('Error:', error.response?.data || error.message);
    });
```

## 🚨 Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid document URL or no questions provided
- **401 Unauthorized**: Invalid or missing authorization token (if enabled)
- **500 Internal Server Error**: Processing errors with detailed messages

Error response format:
```json
{
    "detail": "Error description"
}
```

## 🔒 Security (Optional)

To enable token-based authentication, uncomment these lines in the code:

```python
token = request.headers.get("Authorization", "")
if token != "Bearer your_secret_token":
    raise HTTPException(status_code=401, detail="Unauthorized")
```

Then include the token in requests:
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
     -H "Authorization: Bearer your_secret_token" \
     -H "Content-Type: application/json" \
     -d '{"documents": "...", "questions": [...]}'
```

## 📈 Monitoring and Logs

The application provides detailed console logging:

- 📥 PDF download progress
- ✂️ Chunking statistics
- 🤖 LLM processing status
- ⏱️ Performance metrics
- ❌ Error details with stack traces

## 🔧 Configuration

### Batch Processing Settings

You can adjust these parameters in `utils/llm_chain.py`:

```python
# Batch size for concurrent processing
batch_size = 3  # Adjust based on API rate limits

# Delay between batches
delay = 1.0  # Seconds

# LLM parameters
max_tokens = 300
temperature = 0.1
```

### Chunking Settings

Adjust in `utils/splitter.py`:

```python
chunk_size = 1000      # Maximum chunk size
chunk_overlap = 100    # Overlap between chunks
```

## 🎯 Use Cases

1. **Policy Analysis**: Batch process multiple policy-related questions
2. **Document Q&A**: Extract information from large documents efficiently
3. **Research Automation**: Process research questions in parallel
4. **Customer Support**: Handle multiple customer queries simultaneously
5. **Content Analysis**: Analyze documents with multiple perspectives

## 📞 Support

For issues or questions:
1. Check the console logs for detailed error messages
2. Verify your OpenRouter API key is valid
3. Ensure the document URL is accessible
4. Test with the provided test script

## 🚀 Future Enhancements

- [ ] Support for multiple document formats (Word, Excel, etc.)
- [ ] Alternative PDF processing libraries for better text extraction
- [ ] Caching mechanism for processed documents
- [ ] WebSocket support for real-time updates
- [ ] Advanced context selection algorithms
- [ ] Integration with vector databases for persistent storage

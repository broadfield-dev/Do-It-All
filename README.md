# Do It All
![Banner](do-it-all-im-1.png)

## Overview
The **Do It All** basic agent system for Python. It provides a framework for building automated AI agent scripts in Python using a variety of models and API's.

## Features
- Creates a path and follows it
- Internet Search
- Image Generation
- RAG system
- Document reading
- Normal Chatbot

## Usage

### 1. Clone the repository:
```bash
git clone https://github.com/broadfield-dev/doitall.git
```

### 2. Install dependencies:
```bash
pip install ./doitall
```

### 3. Run the demo script (demo.py):
```bash
python doitall/demo.py
```
### 3.1 Run the demo script (demo2.py):
Demo2 includes RAG, and document reading, and requires additional dependency installs
```bash
pip install ollama langchain langchain_community langchain_chroma langchain_huggingface pypdf
```
Now run the demo
```bash
python doitall/demo2.py
```

# TODO
- Add more model sources
- Screen Reading
- Console Control
- Screen Control
- Voice in/out

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

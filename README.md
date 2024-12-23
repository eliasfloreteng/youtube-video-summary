# Video Summary Generator

A Flask-based web application that generates concise summaries of YouTube videos using their subtitles and Claude 3 AI.

## Features

- YouTube subtitle extraction using yt-dlp
- Automatic text summarization using Claude 3 Haiku
- Rate limiting to prevent API abuse
- Docker support for easy deployment
- Caching of generated summaries
- Parallel processing of large transcripts

## Prerequisites

- Python 3.11+
- Anthropic API key
- Docker (optional)
- FFmpeg

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd youtube-video-summary
```

2. Set up environment:

```bash
# Using virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

3. Set environment variables:

```bash
export ANTHROPIC_API_KEY="your-api-key"
export PORT=8000  # Optional, defaults to 8000
```

## Docker Setup

1. Build the image:

```bash
docker build -t video-summary-generator .
```

2. Run the container:

```bash
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY="your-api-key" \
  -v $(pwd)/uploads:/app/uploads \
  video-summary-generator
```

## Usage

1. Start the server:

```bash
python app.py
```

2. Access the web interface at `http://localhost:8000`

3. Enter a YouTube URL and click "Generate Summary"

## API Endpoints

- `POST /process-video`: Process a new video

  ```json
  {
    "video_url": "https://youtube.com/watch?v=..."
  }
  ```

- `GET /get-summary/<video_id>`: Retrieve cached summary
- `GET /list-summaries`: List all available summaries
- `GET /health`: Health check endpoint

## Rate Limits

- 100 requests per day
- 10 requests per hour
- 2 video processing requests per minute

## Limitations

- Maximum transcript length: 100,000 characters
- English subtitles only
- YouTube videos only
- Requires auto-generated captions

## License

[MIT License](LICENSE)

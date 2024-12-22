from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import yt_dlp
import xml.etree.ElementTree as ET
import os
import html
import logging
import re
from anthropic import Anthropic
import time
from urllib.parse import urlparse, parse_qs
from werkzeug.utils import secure_filename
import concurrent.futures

app = Flask(__name__)

# Add rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per hour"],
    storage_uri="memory://",
)

# Add max transcript length (characters)
MAX_TRANSCRIPT_LENGTH = 100000  # Adjust this value as needed


@app.route("/")
def index():
    return render_template("index.html")


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Anthropic client
anthropic = Anthropic()

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Flask app
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def download_subtitles(url):
    ydl_opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "srv1",
        "noplaylist": True,
        "outtmpl": "%(title)s.%(ext)s",
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # Extract video info first
            info = ydl.extract_info(url, download=False)
            video_title = info.get("title", "Unknown Title")
            logging.info(f"Processing video: {video_title}")

            # Download subtitles
            ydl.download([url])
            logging.info("Subtitle download completed successfully!")

            return video_title

        except Exception as e:
            logging.error(f"An error occurred during download: {str(e)}")
            return None


def clean_text(text):
    if text is None:
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove multiple spaces
    text = " ".join(text.split())

    # Remove special characters and normalize whitespace
    text = text.replace("\n", " ").strip()

    return text


def extract_text_from_srv1(input_file, output_file):
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()

        texts = []
        for text_element in root.findall(".//text"):
            if text_element.text:
                cleaned_text = clean_text(text_element.text)
                if cleaned_text:  # Only add non-empty strings
                    texts.append(cleaned_text)

        # Write to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))

        logging.info(f"Text extracted to: {output_file}")

        # Print first few lines as sample
        sample_lines = texts[:5] if len(texts) > 5 else texts
        logging.info("Sample of processed text:")
        for line in sample_lines:
            logging.info(f"  {line}")

    except ET.ParseError as e:
        logging.error(f"XML parsing error in {input_file}: {str(e)}")
    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}")


def chunk_text(text, chunk_size=4000):
    """Split text into chunks of approximately equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_single_chunk(args):
    """Process a single chunk with Claude API."""
    chunk, index, total = args
    prompt = f"""Please provide a concise summary of this video transcript segment. Focus on the main points and key information:

{chunk}

Provide the summary in a paragraph format."""

    try:
        # Add delay to respect rate limits
        if index > 1:
            time.sleep(1)  # 1 second delay between requests

        response = anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )

        summary = response.content[0].text
        logging.info(f"Processed chunk {index}/{total}")
        return summary

    except Exception as e:
        logging.error(f"Error processing chunk {index}: {str(e)}")
        return f"Error processing chunk {index}"


def summarize_with_claude(text_chunks):
    """Generate summaries for text chunks using Claude 3 Haiku in parallel."""
    summaries = []
    total_chunks = len(text_chunks)

    # Create arguments for each chunk
    chunk_args = [(chunk, i + 1, total_chunks) for i, chunk in enumerate(text_chunks)]

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        summaries = list(executor.map(process_single_chunk, chunk_args))

    return summaries


def process_video_summary(video_url):
    # Download subtitles
    video_title = download_subtitles(video_url)

    if not video_title:
        return

    # Process srv1 files
    srv1_files = [f for f in os.listdir(".") if f.endswith(".srv1")]

    if not srv1_files:
        logging.warning("No .srv1 files found")
        return

    # Process each srv1 file
    for srv1_file in srv1_files:
        # Read the text file
        txt_file = srv1_file.rsplit(".", 1)[0] + ".txt"

        if not os.path.exists(txt_file):
            extract_text_from_srv1(srv1_file, txt_file)

        with open(txt_file, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Check transcript length
        if len(full_text) > MAX_TRANSCRIPT_LENGTH:
            raise ValueError(
                f"Transcript is too long ({len(full_text)} characters). Maximum allowed is {MAX_TRANSCRIPT_LENGTH} characters."
            )

        # Chunk the text
        chunks = chunk_text(full_text)
        logging.info(f"Split text into {len(chunks)} chunks")

        # Generate summaries
        summaries = summarize_with_claude(chunks)

        # Combine summaries and save
        final_summary = "\n\n".join(summaries)
        summary_file = f"{txt_file.rsplit('.', 1)[0]}_summary.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(final_summary)

        logging.info(f"Summary saved to: {summary_file}")

        # Generate final overall summary
        try:
            final_prompt = f"""Based on these segment summaries, provide a concise overall summary of the entire video:

{final_summary}

Please provide a coherent summary that captures the main points and flow of the entire video."""

            final_response = anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.7,
                messages=[{"role": "user", "content": final_prompt}],
            )

            overall_summary = final_response.content[0].text

            # Save overall summary
            overall_summary_file = f"{txt_file.rsplit('.', 1)[0]}_overall_summary.txt"
            with open(overall_summary_file, "w", encoding="utf-8") as f:
                f.write(overall_summary)

            logging.info(f"Overall summary saved to: {overall_summary_file}")

        except Exception as e:
            logging.error(f"Error generating overall summary: {str(e)}")


def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    parsed_url = urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]
        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[2]
        if parsed_url.path.startswith("/v/"):
            return parsed_url.path.split("/")[2]
    return None


def sanitize_video_id(video_id):
    """Sanitize video ID to prevent path traversal."""
    if not video_id:
        return None
    return secure_filename(video_id)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/process-video", methods=["POST"])
@limiter.limit("2 per minute")  # Add specific rate limit for this endpoint
def process_video_endpoint():
    try:
        data = request.get_json()
        if not data or "video_url" not in data:
            return jsonify({"error": "video_url is required"}), 400

        video_url = data["video_url"]
        video_id = extract_video_id(video_url)

        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        # Sanitize video ID
        safe_video_id = sanitize_video_id(video_id)
        if not safe_video_id:
            return jsonify({"error": "Invalid video ID"}), 400

        # Create/check folder for this video
        video_folder = os.path.join(app.config["UPLOAD_FOLDER"], safe_video_id)

        # Check if summary already exists
        summary_file = os.path.join(
            video_folder, f"{safe_video_id}_overall_summary.txt"
        )
        segments_file = os.path.join(
            video_folder, f"{safe_video_id}_chunk_summaries.txt"
        )
        title_file = os.path.join(video_folder, f"{safe_video_id}_title.txt")

        if os.path.exists(summary_file) and os.path.exists(segments_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                overall_summary = f.read()
            with open(segments_file, "r", encoding="utf-8") as f:
                chunk_summaries = f.read().split("\n\n")

            video_title = "Untitled Video"
            if os.path.exists(title_file):
                with open(title_file, "r", encoding="utf-8") as f:
                    video_title = f.read().strip()

            return jsonify(
                {
                    "video_id": safe_video_id,
                    "video_title": video_title,
                    "status": "cached",
                    "results": {
                        "subtitles": {
                            "overall_summary": overall_summary,
                            "chunk_summaries": chunk_summaries,
                        }
                    },
                }
            ), 200

        # If not exists, create folder and process video
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        original_dir = os.getcwd()
        os.chdir(video_folder)

        try:
            # Download subtitles
            video_title = download_subtitles(video_url)

            if not video_title:
                return jsonify({"error": "Failed to download subtitles"}), 500

            # Process srv1 files
            srv1_files = [f for f in os.listdir(".") if f.endswith(".srv1")]

            if not srv1_files:
                return jsonify({"error": "No subtitles found"}), 404

            results = {}

            # Process each srv1 file
            for srv1_file in srv1_files:
                txt_file = srv1_file.rsplit(".", 1)[0] + ".txt"

                if not os.path.exists(txt_file):
                    extract_text_from_srv1(srv1_file, txt_file)

                with open(txt_file, "r", encoding="utf-8") as f:
                    full_text = f.read()

                # Check transcript length
                if len(full_text) > MAX_TRANSCRIPT_LENGTH:
                    return jsonify(
                        {
                            "error": f"Transcript is too long ({len(full_text)} characters). Maximum allowed is {MAX_TRANSCRIPT_LENGTH} characters."
                        }
                    ), 413  # Request Entity Too Large

                # Chunk the text
                chunks = chunk_text(full_text)

                # Generate summaries
                summaries = summarize_with_claude(chunks)

                # Combine summaries
                final_summary = "\n\n".join(summaries)

                # Generate overall summary
                final_prompt = f"""Based on these segment summaries, provide a concise overall summary of the entire video:

                {final_summary}

                Please provide a coherent summary that captures the main points and flow of the entire video."""

                final_response = anthropic.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": final_prompt}],
                )

                overall_summary = final_response.content[0].text

                results[srv1_file] = {
                    "chunk_summaries": summaries,
                    "overall_summary": overall_summary,
                }

            # Update file saving to use safe_video_id
            with open(f"{safe_video_id}_title.txt", "w", encoding="utf-8") as f:
                f.write(video_title)
            with open(
                f"{safe_video_id}_chunk_summaries.txt", "w", encoding="utf-8"
            ) as f:
                f.write(final_summary)
            with open(
                f"{safe_video_id}_overall_summary.txt", "w", encoding="utf-8"
            ) as f:
                f.write(overall_summary)

            return jsonify(
                {
                    "video_id": safe_video_id,
                    "video_title": video_title,
                    "status": "processed",
                    "results": results,
                }
            ), 200

        finally:
            os.chdir(original_dir)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/get-summary/<video_id>", methods=["GET"])
def get_summary(video_id):
    try:
        safe_video_id = sanitize_video_id(video_id)
        if not safe_video_id:
            return jsonify({"error": "Invalid video ID"}), 400

        video_folder = os.path.join(app.config["UPLOAD_FOLDER"], safe_video_id)

        if not os.path.exists(video_folder):
            return jsonify({"error": "Summary not found"}), 404

        results = {}

        summary_files = [
            f"{safe_video_id}_chunk_summaries.txt",
            f"{safe_video_id}_overall_summary.txt",
        ]

        for filename in summary_files:
            filepath = os.path.join(video_folder, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    results[filename] = f.read()

        return jsonify({"video_id": safe_video_id, "summaries": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/list-summaries", methods=["GET"])
def list_summaries():
    try:
        summaries = []
        for video_id in os.listdir(app.config["UPLOAD_FOLDER"]):
            folder_path = os.path.join(app.config["UPLOAD_FOLDER"], video_id)
            if os.path.isdir(folder_path):
                summary_file = os.path.join(
                    folder_path, f"{video_id}_overall_summary.txt"
                )
                if os.path.exists(summary_file):
                    summaries.append(video_id)

        return jsonify({"available_summaries": summaries}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT env var not set
    app.run(debug=True, host="0.0.0.0", port=port)

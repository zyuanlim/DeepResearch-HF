# Open Deep-Research

An open-source implementation of OpenAI's Deep Research agent, designed to perform deep web searches and answer complex questions with high accuracy.

## How It Works

The Deep Research agent is powered by a sophisticated system of tools and components that work together to gather, analyze, and synthesize information:

### Core Components

1. **Web Browser & Search**
   - Uses a text-based browser (similar to Lynx) for web navigation
   - Integrates with Google Search via SerpAPI for comprehensive web searches
   - Can filter search results by year for time-specific queries
   - Maintains browsing history and viewport management

2. **Document Analysis**
   - Handles multiple file formats including:
     - Text documents (PDF, DOCX, TXT)
     - Spreadsheets (XLSX)
     - Presentations (PPTX)
     - Audio files (WAV, MP3)
     - Web pages (HTML)
   - Converts various formats to markdown for consistent processing
   - Implements intelligent text chunking for large documents

3. **Visual Processing**
   - Uses IDEFICS-2 model for image analysis and understanding
   - Supports common image formats (JPEG, PNG)
   - Can generate detailed image descriptions

### Available Tools

1. **Web Navigation Tools**
   - `web_search`: Performs Google searches with optional year filtering
   - `visit_page`: Retrieves and processes webpage content
   - `page_up`/`page_down`: Navigate through long content
   - `find_on_page_ctrl_f`: Search within pages (supports wildcards)
   - `find_next`: Find next occurrence of search terms
   - `find_archived_url`: Access historical versions of web pages via Wayback Machine

2. **Document Processing Tools**
   - `inspect_file_as_text`: Read and analyze various document formats
   - `download_file`: Save files for further analysis
   - Text inspection with automatic summarization for large documents

3. **Visual Analysis Tools**
   - Image processing and captioning
   - Visual question-answering capabilities

### Key Features

- **Memory Management**: Maintains context across multiple searches and page visits
- **Adaptive Processing**: Automatically handles different file formats and content types
- **Error Handling**: Robust error management for failed requests or unsupported formats
- **Content Summarization**: Automatic summarization of large documents
- **Historical Access**: Can access archived versions of web pages
- **Flexible Search**: Supports both broad and time-specific searches

### Integration

The agent is built on top of the `smolagents` framework and can be easily integrated into Gradio applications. It uses environment variables for configuration and supports various API keys for enhanced functionality.

### Example Research Flow

Let's walk through how the agent might research the question: "What were the key findings of the James Webb Space Telescope in 2023?"

1. **Initial Search**
   - Agent uses `web_search` with query "James Webb Space Telescope key discoveries 2023"
   - Filters results to year 2023 using the year filter parameter
   - Receives a list of relevant scientific articles and news sources

2. **Deep Content Analysis**
   - Uses `visit_page` to access the most promising articles
   - For long articles, uses `page_down` to navigate through content
   - Employs `find_on_page_ctrl_f` to locate specific findings and dates
   - If an article references a scientific paper PDF:
     - Uses `download_file` to save the paper
     - Applies `inspect_file_as_text` to analyze its content

3. **Visual Information Processing**
   - When encountering telescope images:
     - Uses visual analysis tools to understand image content
     - Generates descriptions of astronomical features
   - Links visual data with textual findings

4. **Historical Verification**
   - If a link is broken, uses `find_archived_url` to access cached versions
   - Cross-references findings across multiple sources
   - Verifies dates and discoveries through official space agency websites

5. **Information Synthesis**
   - Compiles findings from multiple sources
   - Organizes discoveries chronologically
   - Prioritizes peer-reviewed findings over news articles
   - Generates a comprehensive, fact-checked response

This flow demonstrates how the agent combines multiple tools to:
- Gather information from diverse sources
- Verify findings through cross-referencing
- Process both textual and visual data
- Navigate and extract information from complex documents
- Synthesize a reliable, well-researched answer

The agent can adapt this general flow to research any topic, adjusting its tool usage based on the specific requirements of each query.

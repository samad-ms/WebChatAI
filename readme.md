# WebChatAI

![Python](https://img.shields.io/badge/python-v3.10-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-v3.5-blue)
![Chroma Vector Database](https://img.shields.io/badge/Chroma-Vector%20Database-green)
![LangChain](https://img.shields.io/badge/LangChain-Icon-green)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.0.0-green)

WebChatAI is a Streamlit application that uses LangChain and OpenAI GPT-3.5 Turbo model to create a conversational AI that interacts with websites. This application extracts information from a given website and answers user queries based on the extracted content.

## Features

- **Website Interaction**: Provide a URL, and the application will extract and process the content.
- **Conversational AI**: Ask questions and get answers based on the website's content.
- **History-Aware Retrieval**: The AI takes previous interactions into account to provide more accurate responses.
- **Easy to Use**: User-friendly interface with simple input fields for URLs and queries.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/samad-ms/WebChatAI.git
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your OpenAI API key**:
   Create a `.env` file in the root directory of the project and add your OpenAI API key:
   
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

2. **Use the Application**:
   - Enter the URL of the website you want to interact with in the sidebar.
   - Enter your query in the chat input field.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please feel free to open a pull request or an issue.

## License

This project is licensed under the MIT License.
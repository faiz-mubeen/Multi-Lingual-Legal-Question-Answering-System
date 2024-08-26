# Enhancing Legal Accessibility in India: A Multi-Lingual Legal Question Answering System
![Image](https://github.com/faiz-mubeen/Multi-Lingual-Legal-Question-Answering-System/blob/main/data/legalQna.png)
A multilingual legal question-answering chatbot, designed to assist users who don't have legal knowledge in obtaining answers to their legal queries using the Indian Penal Code (IPC) as a knowledge base. It is multilingual* so anyone from india can understand the answer. 

We've tried out several LLM to do a performance analysis for this task and OpenAI GPT-3.5-turbo-instruct performs pretty well.
![Image](https://github.com/faiz-mubeen/Multi-Lingual-Legal-Question-Answering-System/blob/main/data/perf_comparison.png)
---



---
#### Technologies Used:
- **OpenAI GPT-3.5-turbo-instruct**: Used as the core model for generating legal answers based on IPC context.
- **Chroma**: A vector database for storing embeddings of legal documents and enabling fast retrieval.
- **Streamlit**: A lightweight framework for creating the interactive user interface.
- **Deep Translator (GoogleTranslator)**: Provides translation functionality to return answers in various languages, including Hindi, Urdu, Tamil, and more. We've used three languages for now.

#### Steps of Implementation:
1. **Data Storage and Embedding**: Legal documents related to the IPC are processed and stored as embeddings in Chroma.
2. **Retrieval Chain**: A retrieval chain is created using Langchain, where questions are matched against stored embeddings to retrieve relevant context.
3. **Answer Generation**: Using OpenAI models, answers to legal questions are generated based on retrieved context and are displayed to the user.
4. **Multilingual Translation**: User inputs and answers are translated using GoogleTranslator to allow multilingual communication.

---
#### Dataset:
   The Indian Penal Code (IPC) Book PDF presents a rich and comprehensive dataset that holds immense potential for advancing Natural Language Processing (NLP) tasks and Language Model applications. This dataset encapsulates the entire spectrum of India's criminal law, offering a diverse range of legal principles, provisions, and case laws. With its intricate language and multifaceted legal content, the IPC dataset provides a challenging yet rewarding opportunity for NLP research and development. From text summarization and legal language understanding to sentiment analysis within the context of legal proceedings, this IPC dataset opens avenues for training and fine-tuning Language Models to grasp the nuances of complex legal texts. Leveraging this dataset, researchers and practitioners in the field of NLP can unravel the intricacies of legal discourse and contribute to the advancement of AI-driven legal analysis, interpretation, and decision support systems. [Source](https://huggingface.co/datasets/harshitv804/Indian_Penal_Code)

#### Instructions to Run the App:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have an OpenAI API key:
   - Sign up on [OpenAI](https://beta.openai.com/signup) if you don't have an API key.
   - Set the API key in your environment:
     ```bash
     export OPENAI_API_KEY='your-openai-api-key'
     ```

5. Run the `app.py` file to launch the application:
   ```bash
   streamlit run app.py
   ```

6. Open the local URL provided in the terminal to access the app.

This will set up and run the legal chatbot on your system!

---
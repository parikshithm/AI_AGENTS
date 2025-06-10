
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║              TransGlobal Procurement Automation Agent (v1.0.0)                 ║
# ║                  Author: TransGlobal Tech Team | Last Updated: 2025-03-11      ║
# ║                      License: Proprietary - TransGlobal Industries             ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ OVERVIEW:                                                                      ║
# ║   This application streamlines the entire procurement lifecycle by leveraging  ║
# ║   Retrieval-Augmented Generation (RAG) with Google's Gemini 1.5 Pro LLM. It    ║
# ║   assists in converting business requirements into technical specifications,   ║
# ║   generating RFP documents, matching vendors, crafting tender emails,          ║
# ║   evaluating bids with interactive visualizations, and supporting negotiation  ║
# ║   strategy and risk assessment for contract drafting.                          ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ KEY FEATURES:                                                                  ║
# ║   • Business to Technical Requirements Conversion                              ║
# ║   • Comprehensive RFP Document Generation                                      ║
# ║   • Vendor Matching & Analysis                                                 ║
# ║   • Professional Tender Email Generation                                       ║
# ║   • Bid Evaluation with Interactive Visualizations                             ║
# ║   • Negotiation Strategy Development (incl. BATNA analysis)                    ║
# ║   • Risk Assessment and Contract Drafting                                      ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ ARCHITECTURE:                                                                  ║
# ║   - Frontend: Streamlit with custom CSS styling                                ║
# ║   - NLP Engine: Google Generative AI (Gemini 1.5 Pro)                          ║
# ║   - Vector Database: FAISS for similarity search                               ║
# ║   - Framework: LangChain for RAG implementation                                ║
# ║   - Visualizations: Plotly for interactive charts                              ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ DEPENDENCIES:                                                                  ║
# ║   • Python 3.9+                                                                ║
# ║   • streamlit==1.31.0                                                          ║
# ║   • pandas==2.1.4                                                             ║
# ║   • plotly==5.18.0                                                            ║
# ║   • langchain==0.1.8                                                          ║
# ║   • langchain-google-genai==0.0.7                                             ║
# ║   • faiss-cpu==1.7.4                                                          ║
# ║   • numpy==1.26.3                                                             ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ USAGE:                                                                         ║
# ║   1. Install dependencies: pip install -r requirements.txt                     ║
# ║   2. Set the GOOGLE_API_KEY environment variable                               ║
# ║   3. Run the app: streamlit run procurement-agentv3.py                         ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ DATA SOURCES:                                                                  ║
# ║   • Embedded Procurement Knowledge Base                                        ║
# ║   • Vendor Historical Performance Data                                         ║
# ║   • Sample Bids and RFP Templates                                              ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ WORKFLOW:                                                                      ║
# ║   1. Convert business requirements to technical specifications                 ║
# ║   2. Generate comprehensive RFP documents                                      ║
# ║   3. Match requirements with suitable vendors                                  ║
# ║   4. Create professional tender emails                                         ║
# ║   5. Evaluate and compare vendor bids                                          ║
# ║   6. Develop negotiation strategies (BATNA analysis)                           ║
# ║   7. Assess risks and draft contract elements                                  ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ SECURITY CONSIDERATIONS:                                                       ║
# ║   • API keys are securely managed via environment variables                    ║
# ║   • User data is not stored between sessions                                   ║
# ║   • External API calls are limited (only Google Generative AI is used)         ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ FUTURE IMPROVEMENTS:                                                           ║
# ║   • Integration with persistent databases                                      ║
# ║   • Multi-user support with role-based access                                  ║
# ║   • ERP systems integration                                                    ║
# ║   • Automated document generation in various formats                           ║
# ║   • Machine learning for advanced vendor performance prediction                ║
# ║   • Customizable evaluation criteria weights                                   ║
# ╠════════════════════════════════════════════════════════════════════════════════╣
# ║ CONTACT:                                                                       ║
# ║   For support or feature requests, email:                                      ║
# ║   procurement-support@transglobal-industries.com                               ║
# ╚════════════════════════════════════════════════════════════════════════════════╝



# imported libraries
import os, getpass
import sys
import docx
import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(
    page_title="TransGlobal Procurement Automation",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background with gradient animation */
    .main {
        background: linear-gradient(-45deg, #2b5876, #4e4376, #2b5876, #4e4376);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 20px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* App container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Glass morphism for containers */
    .search-container, .result-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        margin-bottom: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .search-container:hover, .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Futuristic title container */
    .title-container {
        background: linear-gradient(120deg, #000428, #004e92);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 35px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .title-container:before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 100%);
        transform: rotate(45deg);
        z-index: 0;
        animation: shine 5s infinite;
    }
    
    @keyframes shine {
        0% { left: -50%; }
        100% { left: 150%; }
    }
    
    .title-container h1 {
        font-weight: 800;
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Neon buttons */
    .stButton>button {
        background: linear-gradient(90deg, #8E2DE2, #4A00E0);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: bold;
        font-size: 16px;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(138, 43, 226, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(138, 43, 226, 0.6);
        letter-spacing: 1.5px;
    }
    
    .stButton>button:active {
        transform: translateY(1px);
    }
    
    .stButton>button:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transform: translateX(-100%);
        transition: 0.6s;
    }
    
    .stButton>button:hover:before {
        transform: translateX(100%);
    }
    
    /* Card variations with dynamic gradients */
    .vendor-card {
        background: linear-gradient(135deg, rgba(75, 108, 183, 0.05) 0%, rgba(75, 108, 183, 0.15) 100%);
        border-left: 6px solid #4b6cb7;
    }
    
    .rfp-card {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.05) 0%, rgba(76, 175, 80, 0.15) 100%);
        border-left: 6px solid #4CAF50;
    }
    
    .bid-card {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.05) 0%, rgba(33, 150, 243, 0.15) 100%);
        border-left: 6px solid #2196F3;
    }
    
    .contract-card {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.05) 0%, rgba(255, 152, 0, 0.15) 100%);
        border-left: 6px solid #FF9800;
    }
    
    .risk-card {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.05) 0%, rgba(244, 67, 54, 0.15) 100%);
        border-left: 6px solid #F44336;
    }
    
    /* Text inputs and areas */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 12px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #4A00E0;
        box-shadow: 0 0 0 3px rgba(74, 0, 224, 0.15);
    }
    
    /* Tables and dataframes */
    .dataframe {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        width: 100%;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe th {
        background: linear-gradient(90deg, #4b6cb7, #182848);
        color: white;
        padding: 12px;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 14px;
        letter-spacing: 1px;
    }
    
    .dataframe td {
        padding: 12px;
        border-bottom: 1px solid #f0f0f0;
        transition: background-color 0.2s ease;
    }
    
    .dataframe tr:last-child td {
        border-bottom: none;
    }
    
    .dataframe tr:hover td {
        background-color: rgba(74, 0, 224, 0.05);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #4A00E0;
        border-radius: 10px;
        height: 8px;
    }
    
    .stProgress > div > div {
        background-color: rgba(74, 0, 224, 0.2);
        border-radius: 10px;
        height: 8px;
    }
    
    /* Sidebar styling */
    .css-1v3fvcr {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    
    /* Markdown content */
    h2, h3, h4 {
        color: #16213e;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    p, li {
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Custom animated info boxes */
    .stInfo {
        background: linear-gradient(90deg, rgba(33, 150, 243, 0.1), rgba(33, 150, 243, 0.2));
        border-left: 5px solid #2196F3;
        padding: 20px;
        border-radius: 10px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(33, 150, 243, 0); }
        100% { box-shadow: 0 0 0 0 rgba(33, 150, 243, 0); }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #8E2DE2, #4A00E0);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #4A00E0, #8E2DE2);
    }
</style>
""", unsafe_allow_html=True)

# Set environment variables
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Initialize environment
if 'initialized' not in st.session_state:
    _set_env("GOOGLE_API_KEY")
    st.session_state.initialized = True
    st.session_state.current_stage = None
    st.session_state.progress = {}

# Define procurement knowledge documents for FAISS vector store
procurement_documents = [
    # Business to Technical Requirements
    "Business requirements for procurement focus on desired outcomes, while technical requirements detail specific specifications, functionalities, and constraints that suppliers must meet.",
    "Technical requirements should be SMART: Specific, Measurable, Achievable, Relevant, and Time-bound to ensure clear communication with vendors.",
    "Non-functional requirements include performance metrics, security standards, scalability needs, and compliance requirements essential for procurement decisions.",
    
    # RFP Generation
    "An effective RFP includes project overview, technical specifications, vendor qualifications, evaluation criteria, timeline, budget constraints, and submission guidelines.",
    "RFP documents should balance specificity with flexibility to allow vendors to propose innovative solutions while meeting core requirements.",
    "Prompt engineering techniques in RFP generation involve structured questioning, clear constraints, and precise language to ensure comprehensive vendor responses.",
    
    # Vendor Matching
    "Vendor selection criteria include technical capabilities, past performance, financial stability, compliance history, and alignment with project requirements.",
    "Historical vendor data analysis helps identify patterns in delivery reliability, quality consistency, and contract compliance that predict future performance.",
    "Objective vendor ranking methodologies use weighted scoring systems across multiple parameters to mitigate personal bias in the selection process.",
    
    # Tender Email Generation
    "Professional tender emails should include a formal introduction, clear deadlines, submission instructions, contact information, and confidentiality notices.",
    "Email communication for tenders should maintain consistency in branding, tone, and messaging aligned with the organization's communication standards.",
    "Effective tender emails balance information completeness with conciseness to ensure vendors understand requirements without unnecessary complexity.",
    
    # Bid Evaluation
    "Comparative bid analysis techniques include normalized scoring, weighted criteria evaluation, total cost of ownership calculations, and risk-adjusted valuations.",
    "Multi-dimensional bid evaluation frameworks consider pricing structure, implementation timeline, technical compliance, support services, and innovation potential.",
    "Transparent evaluation processes document decision rationale, scoring methodologies, and comparative analyses to support auditable procurement decisions.",
    
    # Negotiation Strategy
    "BATNA (Best Alternative To a Negotiated Agreement) establishes the walkaway threshold and strengthens negotiating position in vendor discussions.",
    "Effective negotiation strategies balance price considerations with value factors including quality, reliability, support, and long-term partnership potential.",
    "Scenario planning for negotiations prepares for multiple potential vendor responses and establishes contingency approaches for various negotiation paths.",
    
    # Risk Assessment and Contract Drafting
    "Comprehensive risk assessments evaluate supplier financial stability, operational reliability, compliance history, geographic risk, and cybersecurity posture.",
    "Contract risk mitigation clauses include performance guarantees, service level agreements, liability provisions, termination rights, and dispute resolution mechanisms.",
    "Effective contracts balance legal protection with partnership building by establishing clear expectations, communication channels, and mutual success metrics."
]

# Initialize the RAG components
@st.cache_resource
def initialize_rag():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_store = FAISS.from_texts(procurement_documents, embeddings)
    retriever = faiss_store.as_retriever(search_kwargs={"k": 5})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.5,
        max_tokens=2000
    )
    
    return retriever, llm

# Generate a prompt template based on the procurement stage
def get_prompt_template(stage):
    templates = {
        "business_to_technical_req": """
        You are an expert procurement automation agent for TransGlobal Industries.
        
        Use the following retrieved context to enhance your analysis:
        {context}
        
        Based on the business requirements provided by the user, convert them into detailed technical requirements.
        
        Business Requirements:
        {question}
        
        Your output must include:
        1. A comprehensive set of functional requirements (specific capabilities and features)
        2. Non-functional requirements (performance, security, scalability, etc.)
        3. Technical specifications with appropriate metrics where possible
        4. Constraints and dependencies that vendors should be aware of
        
        Format your response in markdown with clear sections. Be specific, measurable, and precise.
        """,
        
        "rfp_generation": """
        You are an expert procurement automation agent for TransGlobal Industries.
        
        Use the following retrieved context to enhance your RFP document:
        {context}
        
        Based on the technical requirements provided by the user, generate a complete Request for Proposal (RFP) document.
        
        Technical Requirements:
        {question}
        
        Your RFP must include:
        1. Project overview and objectives
        2. Detailed technical specifications and requirements
        3. Expected deliverables and timeline
        4. Evaluation criteria and vendor qualifications
        5. Submission guidelines and deadlines
        6. Terms and conditions
        
        Format your response as a professional RFP document using markdown. Be comprehensive and avoid ambiguity.
        """,
        
        "vendor_matching": """
        You are an expert procurement automation agent for TransGlobal Industries.
        
        Use the following retrieved context to enhance your vendor analysis:
        {context}
        
        Based on the technical requirements and historical vendor data provided, identify and rank suitable vendors.
        
        Requirements and Vendor Data:
        {question}
        
        Your analysis must include:
        1. Top 3-5 recommended vendors with rankings
        2. Justification for each selection based on historical performance
        3. Alignment between vendor capabilities and project requirements
        4. Potential risks and considerations for each recommended vendor
        
        Format your response in markdown with clear sections. Provide objective analysis without personal bias.
        """,
        
        "tender_email": """
        You are an expert procurement automation agent for TransGlobal Industries.
        
        Use the following retrieved context to enhance your email:
        {context}
        
        Based on the RFP document and vendor information provided, generate a professional email to accompany the tender document.
        
        RFP and Vendor Information:
        {question}
        
        Your email must:
        1. Be professionally formatted with proper salutation and closing
        2. Clearly introduce the procurement opportunity
        3. Provide key dates and submission instructions
        4. Include contact information for queries
        5. Maintain TransGlobal Industries' professional tone and branding
        
        Write a complete, ready-to-send email that is clear, concise, and professional.
        """,
        
        "bid_evaluation": """
        You are an expert procurement automation agent for TransGlobal Industries.
        
        Use the following retrieved context to enhance your bid evaluation:
        {context}
        
        Based on the bid information provided, evaluate and rank the vendor proposals.
        
        Bid Information:
        {question}
        
        Your evaluation must include:
        1. Comparative analysis of all bids received
        2. Scoring for each bid against key criteria (price, quality, timeline, capabilities)
        3. Identification of the top two options with detailed justification
        4. Strengths and weaknesses of each bid
        
        Format your response in markdown with clear sections and tables where appropriate. Provide objective analysis.
        """,
        
        "negotiation_strategy": """
        You are an expert procurement automation agent for TransGlobal Industries.
        
        Use the following retrieved context to enhance your negotiation strategy:
        {context}
        
        Based on the top two bids provided, develop a negotiation strategy including BATNA analysis.
        
        Bid Information:
        {question}
        
        Your strategy must include:
        1. BATNA determination for each negotiation
        2. Key negotiation points and desired outcomes
        3. Potential concessions and their value
        4. Recommended approaches and tactics
        5. Alternative options if negotiations fail
        
        Format your response in markdown with clear, actionable recommendations. Be strategic and thorough.
        """,
        
        "risk_contract": """
        You are an expert procurement automation agent for TransGlobal Industries.
        
        Use the following retrieved context to enhance your risk assessment and contract draft:
        {context}
        
        Based on the vendor and bid information provided, assess risks and draft key contract components.
        
        Vendor and Bid Information:
        {question}
        
        Your output must include:
        1. Comprehensive risk assessment identifying potential issues
        2. Risk mitigation strategies and contingency plans
        3. Key contract clauses to address identified risks
        4. Performance metrics and SLAs to include in the contract
        5. Dispute resolution mechanisms and termination conditions
        
        Format your response in markdown with clear sections. Be thorough and focus on protecting TransGlobal Industries' interests.
        """
    }
    
    return PromptTemplate(
        template=templates.get(stage, templates["business_to_technical_req"]), 
        input_variables=["context", "question"]
    )

# Function to process procurement stage
def process_procurement_stage(stage, input_data, retriever, llm):
    # Create the prompt template for the current stage
    prompt = get_prompt_template(stage)
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Run the chain
    response = qa_chain.run(input_data)
    return response

# Function to analyze vendor data
def analyze_vendor_data(vendor_data):
    # Convert to DataFrame
    df = pd.DataFrame(vendor_data)
    
    # Calculate average scores for each vendor
    vendor_scores = df.groupby('Vendor_name').agg({
        'Delivery_punctuality': 'mean',
        'Quality_of_goods': 'mean',
        'Contract_term_compliance': 'mean'
    }).reset_index()
    
    # Calculate overall score
    vendor_scores['Overall_score'] = (
        vendor_scores['Delivery_punctuality'] + 
        vendor_scores['Quality_of_goods'] + 
        vendor_scores['Contract_term_compliance']
    ) / 3
    
    # Sort by overall score
    vendor_scores = vendor_scores.sort_values('Overall_score', ascending=False)
    
    return vendor_scores

# Function to create vendor performance visualization
def create_vendor_visualization(vendor_scores):
    # Create a radar chart for the top vendors
    top_vendors = vendor_scores.head(5)
    
    fig = go.Figure()
    
    categories = ['Delivery punctuality', 'Quality of goods', 'Contract compliance']
    
    for _, vendor in top_vendors.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[vendor['Delivery_punctuality'], vendor['Quality_of_goods'], vendor['Contract_term_compliance']],
            theta=categories,
            fill='toself',
            name=vendor['Vendor_name']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title="Top Vendor Performance Comparison",
        title_font_size=16,
        height=450,
        width=450
    )
    
    return fig

# Function to analyze bids
def analyze_bids(bid_data):
    # For this simplified version, we'll assume bid_data is already processed
    # In a full implementation, this would parse the bid text into structured data
    
    bids = []
    for bid in bid_data.split("Bid"):
        if not bid.strip():
            continue
        
        bid_info = {"raw_text": "Bid" + bid}
        
        # Extract vendor name
        if ":" in bid.split("\n")[0]:
            bid_info["vendor"] = bid.split("\n")[0].split(":")[1].strip()
        
        # Extract price if available
        price_lines = [line for line in bid.split("\n") if "Total Cost:" in line]
        if price_lines:
            price_text = price_lines[0].strip()
            try:
                # Extract numeric price from string like "Total Cost: $125,000"
                price = price_text.split("$")[1].replace(",", "")
                bid_info["price"] = float(price)
            except:
                bid_info["price"] = 0
        
        # Look for delivery time
        delivery_lines = [line for line in bid.split("\n") if "Lead Time:" in line]
        if delivery_lines:
            bid_info["delivery_time"] = delivery_lines[0].strip()
        
        bids.append(bid_info)
    
    # Sort bids by price
    bids = sorted(bids, key=lambda x: x.get("price", float("inf")))
    
    return bids

# Function to create bid comparison visualization
def create_bid_visualization(bids):
    if not bids or not all(["vendor" in bid and "price" in bid for bid in bids]):
        return None
    
    # Create a bar chart for bid prices
    fig = go.Figure()
    
    vendors = [bid.get("vendor", f"Vendor {i+1}") for i, bid in enumerate(bids)]
    prices = [bid.get("price", 0) for bid in bids]
    
    fig.add_trace(go.Bar(
        x=vendors,
        y=prices,
        marker_color='#4b6cb7',
        text=[f"${p:,.2f}" for p in prices],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Bid Comparison",
        title_font_size=16,
        height=450,
        width=450,
        yaxis=dict(title='Price ($)')
    )
    
    return fig

# Sidebar with app information
with st.sidebar:
    # Enhanced title with icon and styling
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4b6cb7, #182848); 
                padding: 15px; 
                border-radius: 10px; 
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(75, 108, 183, 0.3);">
        <h1 style="color: white; 
                  margin: 0; 
                  text-align: center; 
                  font-size: 1.8rem;
                  text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            🔄 Procurement Automation
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Tool description in a glass-morphism card
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.1); 
                backdrop-filter: blur(5px); 
                padding: 15px; 
                border-radius: 10px; 
                border-left: 4px solid #4b6cb7;
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
        <p style="margin: 0;">This tool automates the procurement process for TransGlobal Industries using advanced AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical overview with animated icons
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(75, 108, 183, 0.05), rgba(24, 40, 72, 0.1)); 
                padding: 15px; 
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);">
        <h3 style="color: #4b6cb7; 
                  margin-top: 0;
                  font-size: 1.2rem;
                  border-bottom: 2px solid rgba(75, 108, 183, 0.2);
                  padding-bottom: 8px;">
            <span style="display: inline-block; animation: pulse 2s infinite;">🌟</span> Advanced Capabilities
        </h3>
        <ul style="list-style-type: none; padding-left: 5px;">
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="color: #4b6cb7; margin-right: 10px;">🤖</span> Powered by Gemini 1.5 Pro LLM
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="color: #4b6cb7; margin-right: 10px;">📚</span> Retrieval-Augmented Generation (RAG)
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="color: #4b6cb7; margin-right: 10px;">🔍</span> FAISS vector similarity search
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="color: #4b6cb7; margin-right: 10px;">📊</span> Interactive Plotly visualizations
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="color: #4b6cb7; margin-right: 10px;">📝</span> Custom prompts for each procurement stage
            </li>
            <li style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="color: #4b6cb7; margin-right: 10px;">🔄</span> End-to-end procurement workflow automation
            </li>
        </ul>
        <p style="font-style: italic; 
                 text-align: center; 
                 margin-top: 15px;
                 font-size: 0.9rem;
                 color: #666;">
            <span style="display: inline-block; animation: float 3s ease-in-out infinite;">🛠️</span> Built with LangChain, FAISS, Streamlit, Google Generative AI, and Plotly
        </p>
    </div>
    
    <style>
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Continue with your existing stages buttons code
    st.subheader("Procurement Stages:")
    
    stages = [
        ("📋 Business to Technical Requirements", "business_to_technical_req"),
        ("📝 RFP Generation", "rfp_generation"),
        ("🔍 Vendor Matching", "vendor_matching"),
        ("📨 Tender Email", "tender_email"),
        ("⚖️ Bid Evaluation", "bid_evaluation"),
        ("🤝 Negotiation Strategy", "negotiation_strategy"),
        ("📊 Risk Assessment & Contract", "risk_contract")
    ]
    
    for stage_name, stage_key in stages:
        if st.button(stage_name, key=f"btn_{stage_key}"):
            st.session_state.current_stage = stage_key
    
    st.subheader("Sample Data")
    
    # Sample business requirements
    if st.button("📋 Load Sample Business Requirements"):
        st.session_state.sample_input = """
        TransGlobal Industries needs to procure 100 high-performance laptops for the engineering department with the following business requirements:
        1. The laptops must support complex CAD software and engineering applications
        2. They should have sufficient battery life for day-long use
        3. They must be compatible with our existing network infrastructure
        4. The procurement must be completed within the next 8 weeks
        5. The budget is approximately $120,000 for the entire purchase
        6. We require a minimum 3-year warranty with onsite support
        """
        
# Main content
st.markdown("<div class='title-container'><h1>TransGlobal Industries Procurement Automation</h1><p>Advanced AI-Powered Procurement Process</p></div>", unsafe_allow_html=True)

# Initialize RAG components
retriever, llm = initialize_rag()

# Load vendor data
@st.cache_data
def load_vendor_data():
    # This would typically load from a CSV file
    # For simplicity, we're creating a sample dataframe
    data = []
    for vendor in ["Vendor A", "Vendor B", "Vendor C", "Vendor D", "Vendor E"]:
        for _ in range(10):
            data.append({
                "Vendor_name": vendor,
                "Interaction_date": "2024-01-01",
                "Delivery_punctuality": np.random.randint(1, 11),
                "Quality_of_goods": np.random.randint(1, 11),
                "Contract_term_compliance": np.random.randint(1, 11),
                "Comments": np.random.choice(["Exceeded expectations", "Met expectations", "Minor issues", "Below expectations"])
            })
    return pd.DataFrame(data)

# Import numpy for the vendor data generation
import numpy as np
vendor_data = load_vendor_data()

# Current stage view
if st.session_state.current_stage:
    stage_titles = {
        "business_to_technical_req": "Business to Technical Requirements Conversion",
        "rfp_generation": "RFP Generation",
        "vendor_matching": "Vendor Matching",
        "tender_email": "Tender Email Generation",
        "bid_evaluation": "Bid Evaluation",
        "negotiation_strategy": "Negotiation Strategy & BATNA Analysis",
        "risk_contract": "Risk Assessment & Contract Drafting"
    }
    
    current_stage = st.session_state.current_stage
    st.markdown(f"## {stage_titles[current_stage]}")
    
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    
    # Input area based on current stage
    if current_stage == "business_to_technical_req":
        input_data = st.text_area(
            "Enter Business Requirements:",
            height=200,
            value=st.session_state.get('sample_input', ''),
            placeholder="Describe the business needs and requirements for the procurement..."
        )
        
    elif current_stage == "rfp_generation":
        if "business_to_technical_req" in st.session_state.progress:
            st.info("Using technical requirements from previous stage. You can edit if needed.")
            input_data = st.text_area(
                "Technical Requirements for RFP:",
                height=200,
                value=st.session_state.progress["business_to_technical_req"],
                placeholder="Technical requirements for generating RFP..."
            )
        else:
            input_data = st.text_area(
                "Enter Technical Requirements:",
                height=200,
                placeholder="Enter the technical specifications and requirements for the RFP..."
            )
            
    elif current_stage == "vendor_matching":
        if "rfp_generation" in st.session_state.progress:
            st.info("Using RFP from previous stage. You can edit if needed.")
            col1, col2 = st.columns(2)
            with col1:
                input_data = st.text_area(
                    "RFP Document:",
                    height=300,
                    value=st.session_state.progress["rfp_generation"],
                    placeholder="Enter RFP details for vendor matching..."
                )
            with col2:
                st.write("Vendor Historical Performance:")
                st.dataframe(vendor_data.head(10))
        else:
            col1, col2 = st.columns(2)
            with col1:
                input_data = st.text_area(
                    "Enter RFP Document:",
                    height=300,
                    placeholder="Enter RFP details for vendor matching..."
                )
            with col2:
                st.write("Vendor Historical Performance:")
                st.dataframe(vendor_data.head(10))
                
    elif current_stage == "tender_email":
        if "vendor_matching" in st.session_state.progress and "rfp_generation" in st.session_state.progress:
            st.info("Using RFP and vendor selection from previous stages. You can edit if needed.")
            input_data = f"""
            RFP Document:
            {st.session_state.progress["rfp_generation"]}
            
            Selected Vendors:
            {st.session_state.progress["vendor_matching"]}
            """
            input_data = st.text_area(
                "RFP and Vendor Information:",
                height=300,
                value=input_data,
                placeholder="Enter RFP and vendor information for email generation..."
            )
        else:
            input_data = st.text_area(
                "Enter RFP and Vendor Information:",
                height=300,
                placeholder="Enter RFP and vendor information for email generation..."
            )
            
    elif current_stage == "bid_evaluation":
        # Load sample bid data
        if st.checkbox("Use Sample Bid Data"):
            sample_bids = """
            Bid 1: Tech Solutions Ltd.
            Technical Proposal:
            Processor: Intel Core i7 (11th Gen)
            RAM: 16GB DDR4
            Storage: 512GB SSD
            Financial Proposal:
            Unit Price: $1,200 per laptop
            Total Cost: $125,000
            
            Bid 2: Apex Computers
            Technical Proposal:
            Processor: AMD Ryzen 7 (5000 series)
            RAM: 16GB DDR4
            Storage: 1TB SSD
            Financial Proposal:
            Unit Price: $1,150 per laptop
            Total Cost: $119,000
            
            Bid 3: Digital Edge
            Technical Proposal:
            Processor: AMD Ryzen 7 (4000 series)
            RAM: 16GB DDR4
            Storage: 512GB SSD
            Financial Proposal:
            Unit Price: $1,100 per laptop
            Total Cost: $114,500
            """
            input_data = st.text_area(
                "Bid Information:",
                height=300,
                value=sample_bids,
                placeholder="Enter bid information for evaluation..."
            )
        else:
            input_data = st.text_area(
                "Enter Bid Information:",
                height=300,
                placeholder="Enter bid information for evaluation..."
            )
    
    elif current_stage == "negotiation_strategy":
        if "bid_evaluation" in st.session_state.progress:
            st.info("Using top bids from previous stage. You can edit if needed.")
            input_data = st.text_area(
                "Top Bids for Negotiation:",
                height=300,
                value=st.session_state.progress["bid_evaluation"],
                placeholder="Enter top bid information for negotiation strategy..."
            )
        else:
            input_data = st.text_area(
                "Enter Top Bids:",
                height=300,
                placeholder="Enter top bid information for negotiation strategy..."
            )
            
    elif current_stage == "risk_contract":
        if "negotiation_strategy" in st.session_state.progress and "bid_evaluation" in st.session_state.progress:
            st.info("Using negotiation strategy and bid information from previous stages. You can edit if needed.")
            input_data = f"""
            Bid Information:
            {st.session_state.progress["bid_evaluation"]}
            
            Negotiation Strategy:
            {st.session_state.progress["negotiation_strategy"]}
            """
            input_data = st.text_area(
                "Vendor and Bid Information:",
                height=300,
                value=input_data,
                placeholder="Enter vendor and bid information for risk assessment and contract drafting..."
            )
        else:
            input_data = st.text_area(
                "Enter Vendor and Bid Information:",
                height=300,
                placeholder="Enter vendor and bid information for risk assessment and contract drafting..."
            )
    
    # Process button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        process_button = st.button("🔄 Process", key="process_btn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process the current stage
    if process_button and input_data:
        with st.spinner(f"Processing {stage_titles[current_stage]}..."):
            # Add a slight delay and animation for better UX
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Process the stage
            result = process_procurement_stage(current_stage, input_data, retriever, llm)
            st.session_state.progress[current_stage] = result
            
            # Display success message
            st.success(f"{stage_titles[current_stage]} completed successfully!")
    
    # Display results if available
    if current_stage in st.session_state.progress:
        result = st.session_state.progress[current_stage]
        
        if current_stage == "business_to_technical_req":
            st.markdown("<div class='result-card rfp-card'>", unsafe_allow_html=True)
            st.markdown("### Technical Requirements")
            st.markdown(result)
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif current_stage == "rfp_generation":
            st.markdown("<div class='result-card rfp-card'>", unsafe_allow_html=True)
            st.markdown("### Request for Proposal (RFP)")
            st.markdown(result)
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif current_stage == "vendor_matching":
            st.markdown("<div class='result-card vendor-card'>", unsafe_allow_html=True)
            st.markdown("### Vendor Selection Results")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(result)
            
            with col2:
                # Create vendor visualization
                vendor_scores = analyze_vendor_data(vendor_data)
                fig = create_vendor_visualization(vendor_scores)
                st.plotly_chart(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif current_stage == "tender_email":
            st.markdown("<div class='result-card rfp-card'>", unsafe_allow_html=True)
            st.markdown("### Tender Email")
            st.markdown(result)
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif current_stage == "bid_evaluation":
            st.markdown("<div class='result-card bid-card'>", unsafe_allow_html=True)
            st.markdown("### Bid Evaluation Results")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(result)
            
            with col2:
                # Create bid visualization
                bids = analyze_bids(input_data)
                if bids:
                    fig = create_bid_visualization(bids)
                    if fig:
                        st.plotly_chart(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif current_stage == "negotiation_strategy":
            st.markdown("<div class='result-card bid-card'>", unsafe_allow_html=True)
            st.markdown("### Negotiation Strategy & BATNA")
            st.markdown(result)
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif current_stage == "risk_contract":
            st.markdown("<div class='result-card risk-card'>", unsafe_allow_html=True)
            st.markdown("### Risk Assessment & Contract Elements")
            st.markdown(result)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # # Navigation to next stage if available
        # if current_stage != "risk_contract":  # If not the last stage
        #     stage_keys = [s[1] for s in stages]
        #     current_index = stage_keys.index(current_stage)
        #     next_stage = stage_keys[current_index + 1]
        #     next_stage_name = [s[0] for s in stages if s[1] == next_stage][0]
            
        #     col1, col2, col3 = st.columns([1, 1, 1])
        #     with col2:
        #         if st.button(f"Next: {next_stage_name}",

# Adding a section to display library versions
def display_version_info():
    # Create a collapsible section for version information
    with st.expander("System Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            
            st.text(f"Python: {sys.version.split()[0]}")
            st.text(f"Streamlit: {st.__version__}")
        
        with col2:
            st.write("**Libraries:**")
            st.text(f"LangChain: {__import__('langchain').__version__}")
            # st.text(f"LangChain Google GenAI: {__import__('langchain_google_genai').__version__}")
            st.text(f"FAISS: {__import__('faiss').__version__}")
            st.text(f"PyPDF2: {__import__('PyPDF2').__version__}")
            st.text(f"Docx: {docx.__version__}")

display_version_info()        
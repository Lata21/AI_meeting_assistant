import streamlit as st
import os
import tempfile
import pandas as pd
from datetime import date

# Loaders for PDFs and text files
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Splitting text
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLM from Groq
from langchain_groq import ChatGroq

# Open-source embeddings (since Groq does not support embeddings)
from langchain.embeddings import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# QA chain
from langchain.chains import RetrievalQA

# Prompt template (used optionally with chains or tasks)
from langchain.prompts import PromptTemplate

# CrewAI task orchestration
from crewai import Agent, Task, Crew
from crewai.process import Process

from utils import TodoistTools, TranscriptExtractor, TelegramCommunicator, TaskExtractor, TodoistMeetingManager


def initialiaze_session_state():
    """Initialize all session state variables"""
    defaults = {
        "setup": None,
        "groq_api_key": "",
        "prepared": False,
        "vectorstore": None,
        "context_analysis": None,
        "meeting_strategy": None,
        "executive_brief": None,
        "todoist_api_key": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "transcript_source": "google_meet",
        "meeting_id": "",
        "todoist_manager": None,
        "task_extraction_results": None
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def create_vectorspace(docs):
    """Create a vector store from documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)


def create_qa_chain(vectorstore, api_key):
    """Create a QA chain for answering questions"""
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following content to answer the question.
If you don't know the answer, say that you don't know.

Context: {context}
Question: {question}

Answer:"""
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7, groq_api_key=api_key)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )



def run_crewai_analysis(setup, llm):
    """Run CrewAI Analysis for meeting preparation"""
    attendees_text = "\n".join([f"- {attendee}" for attendee in setup['attendees']])

    context_agent = Agent(
        role='Content Analyst',
        goal='Provide comprehensive context analysis for the meeting',
        backstory="""You are an expert business analyst who specializes in preparing documents for meetings.
        You thoroughly research companies and identify key stakeholders.""",
        llm=llm,
        verbose=True
    )

    strategy_agent = Agent(
        role='Meeting Strategist',
        goal='Create detailed meeting strategy and agenda',
        backstory="""You are a seasoned meeting facilitator who excels at
        structuring effective business discussions. You understand how to allocate time optimally.""",
        llm=llm,
        verbose=True
    )

    brief_agent = Agent(
        role='Executive Briefer',
        goal='Generate executive briefing with actionable insights',
        backstory="""You are a master communicator who specializes in crafting
        executive briefings. You distill complex information into clear, concise documents.""",
        llm=llm,
        verbose=True
    )

    context_task = Task(
        description=f"""Analyze the content for the meeting with {setup['company']}.
Consider:
1. Company background and market position
2. Meeting Objective: {setup['objective']}
3. Attendees: {attendees_text}
4. Focus areas: {setup['focus']}

FORMAT IN MARKDOWN with clear headings.""",
        agent=context_agent,
        expected_output="""A markdown-formatted content analysis with sections for:
- Executive Summary
- Company Background
- Situation Analysis
- Key Stakeholders
- Strategic Considerations"""
    )

    strategy_task = Task(
        description=f"""Develop a structured meeting strategy for the upcoming discussion with {setup['company']}.
Include:
1. Clear agenda with estimated time allocations
2. Key topics to cover based on the meeting objective: {setup['objective']}
3. Recommended order of discussion
4. Specific questions or insights to gather from attendees:
   {attendees_text}
5. Strategic framing of focus areas: {setup['focus']}

FORMAT IN MARKDOWN with clear sections and bullet points.""",
        agent=strategy_agent,
        expected_output="""A markdown-formatted meeting strategy including:
- Meeting Objective
- Structured Agenda
- Time Allocations
- Key Questions
- Strategic Framing"""
    )

    brief_task = Task(
        description=f"""Generate a concise executive briefing for the meeting with {setup['company']}.
Focus on:
1. Key meeting goals
2. Executive summary of important context and focus points
3. Actionable insights for decision-makers
4. Attendee-specific considerations: {attendees_text}
5. Suggested next steps post-meeting

FORMAT IN MARKDOWN suitable for C-level executives.""",
        agent=brief_agent,
        expected_output="""A markdown-formatted executive brief including:
- Executive Summary
- Meeting Goals
- Key Insights
- Stakeholder Considerations
- Recommended Actions"""
    )

    crew = Crew(
        agents=[context_agent, strategy_agent, brief_agent],
        tasks=[context_task, strategy_task, brief_task],
        verbose=True,
        process=Process.sequential
    )

    return crew.kickoff()

def extract_content(result_item):
    """Extract content from CrewAI result item"""
    if hasattr(result_item, 'result'):
        return result_item.result
    if isinstance(result_item, dict) and 'result' in result_item:
        return result_item['result']
    if isinstance(result_item, str):
        return result_item
    return str(result_item)


def fallback_analysis(setup, llm):
    """Fallback method if CrewAI fails"""
    attendees_text = "\n".join([f"- {attendee}" for attendee in setup['attendees']])

    # Context Analysis Prompt
    context_prompt = f"""Analyze the content for the meeting with {setup['company']}:

- Meeting objective: {setup['objective']}
- Attendees:
{attendees_text}
- Focus areas: {setup['focus']}

Format in markdown with appropriate headings like:
- Executive Summary
- Company Background
- Situation Analysis
- Key Stakeholders
- Strategic Considerations
"""

    # Strategy Prompt
    strategy_prompt = f"""Develop a structured meeting strategy for the upcoming discussion with {setup['company']}:

- Meeting objective: {setup['objective']}
- Attendees:
{attendees_text}
- Focus areas: {setup['focus']}

Include:
1. A clear agenda with time estimates
2. Key topics aligned with the meeting objective
3. Recommended sequence of discussion
4. Strategic questions to ask attendees
5. Framing for focus areas

Format in markdown with headings and bullet points like:
- Meeting Objective
- Agenda
- Time Allocations
- Discussion Points
- Key Questions
"""

    # Brief Prompt
    brief_prompt = f"""Create a concise executive briefing for the meeting with {setup['company']}:

- Meeting objective: {setup['objective']}
- Attendees:
{attendees_text}
- Focus areas: {setup['focus']}

Focus on:
1. Summary of the meeting purpose
2. Key contextual insights
3. Attendee-specific considerations
4. Strategic recommendations
5. Clear next steps

Format in markdown suitable for executives with headings like:
- Executive Summary
- Meeting Goals
- Key Insights
- Stakeholder Considerations
- Recommended Actions
"""

    # Call LLM (adjust method based on actual usage, e.g., `llm.invoke()` or `llm.predict()`)
    context_content = llm.invoke(context_prompt).content
    strategy_content = llm.invoke(strategy_prompt).content
    brief_content = llm.invoke(brief_prompt).content

    return context_content, strategy_content, brief_content




def create_qa_chain(vectorstore,api_key):
    """Createb a QA chain for answering questions"""
    prompt_template = PromptTemplate(
        input_variables = ["context","question"],
        template = """Use the following content to answer the question.
        If you don't know the answer , say that you don't know.
        
        Context : {context}
        Question :{question}
        
        Answer: """
    )

    retriver = vectorstore.as_retriver(search_krwags={"k":3})

    return RetrievalQA.from_chain_type(
        llm = HuggingFaceEmbeddings(model="llama3-70b-8192",temperature = 0.7 ,api_key=api_key),
        chain_type ="stuff",
        retriver = retriver,
        chain_type_krwags ={"prompt":prompt_template},
        return_source_documents = True
    )


def send_telegram_notification(telegram_bot_token, telegram_chat_id, results):
    """Send notification about tasks to Telegram"""
    # Initialize Telegram communicator
    telegram = TelegramCommunicator(telegram_bot_token, telegram_chat_id)

    # Create summary message
    summary = "*📝 Meeting Task Summary*\n\n"
    summary += f"*📂 Projects:* {', '.join(results['projects_created'])}\n\n"
    summary += f"*✅ Tasks Created:* {len(results['tasks_created'])}\n\n"

    for task in results["tasks_created"]:
        summary += f"• {task['content']} (Project: {task['project']})\n"
        if task.get("assignee"):
            summary += f"  └ Assigned to: {task['assignee']}\n"
        summary += "\n"

    # Send message
    return telegram.send_message(summary)


def process_transcript(transcript, todoist_manager):
    """Process a transcript and extract tasks."""

    task_extractor = TaskExtractor(todoist_manager.task_extractor.llm)
    extracted_data =  task_extractor.extract_tasks_from_transcript(transcript)
    if "error" in extracted_data:
        return {"error": extracted_data["error"]}
    
    results = {
        "project_created":[],
        "tasks_created":[]
    }
    for project_data in extracted_data.get('projects'):
        project_name = project_data["name"]
        project = todoist_manager.todoist_tools.create_project(project_name)
        
        if "error" in project:
            results["error"] = project["error"]
            return results
        

        results["project_created"].append(project_name)

        for task_data in project_data["tasks"]:
            task = todoist_manager.todoist_tools.create_and_assign_task(
                task_data["content"],
                project_name,
                task_data.get("assignee"),
                task_data.get("due_string"),
                task_data.get("priority",3)
            )

            if "error" in task:
                results["task_errors"] = results.get("task_errors",[]) + [task["error"]]

            else:
                results["tasks_created"].append({
                    "content": task_data["content"],
                    "project": project_name,
                    "assignee": task_data.get("assignee")

                })
        return results


def main():
    st.set_page_config(page_title="AI Meeting Assistant", page_icon="📝", layout="wide")
    st.title("📝 AI Meeting Assistant")

    # Initialize session state (assuming you have a function to do this)
    initialiaze_session_state()

    with st.sidebar:
        # OpenAI API Key
        openai_api_key = st.text_input("GROQ API KEY", type="password", value=st.session_state["groq_api_key"])
        if openai_api_key:
            st.session_state["groq_api_key"] = "groq_api_key"
            os.environ["GROQ_API_KEY"] = "groq_api_key"

        # Todoist API Key
        todoist_api_key = st.text_input("Todoist API KEY", type="password", value=st.session_state["todoist_api_key"])
        if todoist_api_key != st.session_state["todoist_api_key"]:
            st.session_state["todoist_api_key"] = todoist_api_key
            st.session_state["todoist_manager"] = None

        # Telegram Integration
        with st.expander("Telegram Integration (Optional)"):
            telegram_bot_token = st.text_input("Telegram Bot Token", type="password", value=st.session_state["telegram_bot_token"])
            telegram_chat_id = st.text_input("Telegram Chat ID", value=st.session_state["telegram_chat_id"])

            telegram_credentials_changed = False
            if telegram_bot_token != st.session_state["telegram_bot_token"]:
                st.session_state["telegram_bot_token"] = telegram_bot_token
                telegram_credentials_changed = True

            if telegram_chat_id != st.session_state["telegram_chat_id"]:
                st.session_state["telegram_chat_id"] = telegram_chat_id
                telegram_credentials_changed = True

            if telegram_credentials_changed and st.session_state["todoist_api_key"]:
                st.session_state["todoist_manager"] = None

        st.info("Boost your meeting productivity with automated context analysis, smart agenda planning, and integrated task management.")

    tab_setup, tab_results, tab_qa, tab_task = st.tabs(["Meeting Setup", "Preparation Results", "Q&A Assistant", "Task Management"])
# ======= Meeting Setup Tab =======
    
    with tab_setup:
        st.subheader("Meeting Configuration")
        company_name = st.text_input("Company Name")
        meeting_objective = st.text_area("Meeting Objective")
        meeting_date = st.date_input("Meeting Date", value = date.today())
        meeting_duration = st.slider("Meeting Duration (minutes)", 15, 180, 60)

        st.subheader("Attendees")
        attendees_data = st.data_editor(
            pd.DataFrame({"Name": [""], "Role": [""], "Company": [""]}),
            num_rows="dynamic",
            use_container_width=True
        )

        focus_areas = st.text_area("Focus Area or Concerns")

        uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["txt", "pdf"])

        if st.button("Prepare Meeting", type="primary", use_container_width=True):
            if not openai_api_key or not company_name or not meeting_objective:
                st.error("Please fill all required details.")
            else:
                attendees_formatted = []
                for _, row in attendees_data.iterrows():
                    if row["Name"]:
                        attendees_formatted.append(f"{row['Name']}, {row['Role']}, {row['Company']}")

                st.session_state["setup"] = {
                    "company": company_name,
                    "objective": meeting_objective,
                    "date": meeting_date,
                    "duration": meeting_duration,
                    "attendees": attendees_formatted,
                    "focus": focus_areas,
                    "files": uploaded_files
                }

                st.session_state["prepared"] = False
                st.rerun()

    # ======= Results Tab =======
    with tab_results:
        if st.session_state.get("setup") and not st.session_state.get("prepared"):
            with st.status("Processing meeting data...", expanded=True) as status:
                setup = st.session_state["setup"]

                attendees_text = "\n".join([f"- {attendee}" for attendee in setup['attendees']])
                base_context = f"""
Meeting Information:
- Company: {setup['company']}
- Objective: {setup['objective']}
- Date: {setup['date']}
- Duration: {setup['duration']} minutes
- Focus Area: {setup['focus']}

Attendees:
{attendees_text}
"""
# Process the document

                docs = process_documents(base_context,setup['files'])

                #Create vector store
                vectorstore = create_vectorspace(docs)
                st.session_state["vectorstore"] = vectorstore

                #Initalize the LLM
                llm = ChatGroq(model="llama3-70b-8192",temperature=0.7,api_key =st.session_state["groq_api_key"])
                try:
                    result = run_crewai_analysis(setup,llm)

                    if isinstance(result,list) and len(result)>=3:
                        context_content = extract_content(result[0])
                        strategy_content = extract_content(result[1])
                        brief_content  = extract_content(result[2])
                    else:
                        raise Exception("CrewAI did not return expected format")
                except Exception as e:
                    st.warning(f"Using fallback method. Error : {str(e)}")
                    context_content, strategy_content, brief_content = fallback_analysis(setup , llm) 


                st.session_state.update({
                    "context_analysis": context_content,
                    "meeting_strategy": strategy_content,
                    "executive_brief": brief_content,
                    "prepared": True

                })

                status.update(label = "Meeting preparation complete!",
                              state="complete",expanded = False)


        if st.session_state["prepared"]:
            result_tab1 , result_tab2 , result_tab3 =  st.tabs(["Context Analysis","Meeting Strategy","Executive Brief"])


            with result_tab1:
                if st.session_sate["context_analysis"]:
                    st.markdown(st.session_state["context_analysis"])
                else:
                    st.warning("Context analysis not generated")

            with result_tab2:
                if st.session_sate["meeting_strategy"]:
                    st.markdown(st.session_state["meeting_strategy"])
                else:
                    st.warning("Meeting strategy not generated")

            with result_tab3:
                if st.session_sate["executive_brief"]:
                    st.markdown(st.session_state["executive_brief"])
                else:
                    st.warning("Executive brief not generated")
        

            col1, col2 , col3 = st.columns(3)
            with col1:
                if st.session_state["context_analysis"]:
                    st.download_button("Download Context Analysis",st.session_state["context_analysis"],
                                   "context_analysis.md",
                                   use_container_width = True)
        
        
            with col2:
                if st.session_state["meeting_startegy"]:
                    st.download_button("Download Meeting Startegy",st.session_state["meeting_startegy"],
                                   "meeting_startegy.md",
                                   use_container_width = True)
                

            with col3:
                if st.session_state["executive_brief"]:
                    st.download_button("Download Executive Brief",st.session_state["executive_brief"],
                                   "executive_brief.md",
                                   use_container_width = True)


        else:
            st.info("Please Configure your meeting in the 'Meeting Setup ")            

        
        
    with tab_qa:
        st.subheader("Meeting Q&A Assistant")

        if not st.session_state["groq_api_key"]:
            st.warning("Please enter your GROQ API key in the sidebar.")
        elif st.session_state["vectorstore"] is None:
            st.info("Please prepare a meeitng first to use the Q&A feature.")
        
        else:
            st.success("Ask questions about your meeting below:")

            question = st.text_input("Your question:",key ="qa_question")

            if question:
                with st.spinner("Finding answer...."):
                    try:
                        qa = create_qa_chain(st.session_state["vectorstore"],st.session_state["groq_api_key"])

                        result = qa.invoke({"query":question})

                        st.markdown('##### Answer')
                        st.markdown(result["result"])

                        #Show sources
                        with st.expander("View Source Documnets"):
                            for i, doc in enumerate(result.get("source_documents",[])):
                                st.markdown(f"**Source {i+1}**")
                                st.markdown(f"```\n{doc.page_content}\n```")
                                st.divider()

                    except Exception as e :
                        st.error(f"Error: {str(e)}")
                        st.error("Please check your question and try again.")



    with tab_task:
        st.subheader("Meeting Task Management")

        if not st.session_state["groq_api_key"]:
            st.warning("Please enter yout Groq API key in the sidebar.")
        elif not st.session_state["todoist_api_key"]:
            st.warning("Please enter the Todoist API Key in the sidebar.")
        else:
            if st.session_state["todoist_manager"] is None:
                llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.3,
    groq_api_key=st.session_state["groq_api_key"]  # store your Groq key here
)


                st.session_state["todoist_manger"] = TodoistMeetingManager(
    st.session_state["todoist_api_key"],
    st.session_state["telegram_bot_token"] or None,
    st.session_state["telegram_chat_id"] or None,
    st.session_state["transcript_source"] or None,
    llm
)

            col1, col2 = st.columns([1,2])
            with col1:
                transcript_source = st.selectbox("Trascript Source",index =["google_meet","whatsapp","telegram"].index(st.session_state["trascript_source"]))

                if transcript_source != st.session_state["transcript_source"]:
                    st.session_state["transcript_source"] = transcript_source

                    st.session_state["todoist_manager"].transcript_extractor = TranscriptExtractor(transcript_source)

            
            with col2:
                meeting_id = st.text_input("Meeting/Chat ID",
                                           value = st.session_state["meeting_id"],
                                           help = "Enter the ID of your Google meet , Whatsapp or Telegram Conversation")
                
                if meeting_id != st.session_state["meeting_id"]:
                    st.session_state["meeting_id"] = meeting_id
            
            with st.expander("Manual Transcript Input (Optional)"):
                manual_transcript = st.text_area(
                    "Enter Meeting Transcript",
                    height = 200,
                    help = "If you don't have API access , you can manually paste a transcript here."
                )
            
            if st.button("Extract Tasks from Meeting", type ="primary",use_container_width =True):
                if not meeting_id and not manual_transcript:
                    st.error("Please provide either Meeting ID or Manual Transcript.")
                else:
                    with st.spinner("Processing the transcript and extracting tasks....."):
                        try:

                            if manual_transcript:
                                results = process_transcript(manual_transcript,st.session_state["todoist_manager"]
                                )
                            else:
                                results = st.session_state["todoist_manager"].process_meeting(meeting_id)
                            
                            st.session_state["task_extraction_results"] = results
                        
                        except Exception as e:
                            st.error(f"Error processing meeting : {str(e)}")

                    if st.session_state["task_extraction_results"]:
                        results = st.session_state["task_extraction_results"]

                        if "error" in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.success("Meeting Processed Successfullt!!")

                            if results["projects_created"]:
                                st.subheader("Project Created")
                                for project in results["projects_created"]:
                                    st.write(f"-{project}")
                            

                            if results["tasks_created"]:
                                st.suheader("Tasks Created")
                                task_df = pd.DataFrame(results["tasks_created"])
                                st.dataframe(task_df)
                            

                            if "tasks_error" in results  and results["task_errors"]:
                                with st.expander("Task Creation Errors"):
                                    for error in results("task_errors"):
                                        st.error(error)


                            if st.session_state["telegram_bot_token"] and st.session_state["telegram_chat_id"]:
                                if st.button("Notify Team on Telegram"):
                                    with st.spinner("Sending Notification...."):
                                        try:
                                            message_result = send_telegram_notification(
                                                st.session_state["telegram_bot_token"],
                                                st.session_state["telegram_chat_id"],
                                                results

                                            )
                                            if "error" in message_result:
                                                st.error(f"Error Sending Notification :{message_result['error']}")
                                            else:
                                                st.success("Team notification sent successfully!")
                                        except Exception as e:
                                            st.error(f"Error sending notification : {str(e)}")

                            with st.expander("Manage Todoist Projects and Tasks"):
                                if st.button("Refresh Projects"):
                                    try:
                                        projects =  st.session_state["todoist_manager"].todoist_tools.get_projects()
                                        if "error" in projects:
                                            st.error(f"Error fetching projects: {projects['error']}")

                                        else:
                                            project_df = pd. DataFrame([{"id": p["id"],"name": p["name"]} for p in projects])
                                            st.dataframe(project_df)
                                    except Exception as e:
                                        st.error(f"Error fetching projects : {str(e)}")
                                


if __name__ == "__main__":
    main()
                                     
                                                         

                    










        
                


            
            







            



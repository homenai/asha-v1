import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'genai_client' not in st.session_state:
        try:
            # Set up Gemini client
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise KeyError("GOOGLE_API_KEY not found in environment variables")
            
            # Initialize the chat model
            st.session_state.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.7,
                convert_system_message_to_human=True
            )
        except Exception as e:
            st.error(f"""
                Error initializing Gemini client: {str(e)}
                Please ensure GOOGLE_API_KEY is set in your .env file.
                """)
            st.stop()

def generate_dynamic_prompt(history):
    """Generate a dynamic system prompt based on the conversation history."""
    base_prompt = """
    Role & Qualities

    You are an AI focused on supportive listening and emotional well-being
    Empathetic, non-judgmental, patient, and attentive
    Gentle but clear communication, warm yet professional
    Guidelines

    Actively listen and reflect feelings
    Validate emotions before offering solutions
    Ask open-ended questions to encourage sharing
    Offer coping strategies when appropriate
    Use affirming phrases (e.g., "I hear you")
    Maintain professional boundaries
    Safety Protocol

    If user mentions self-harm, suicidal thoughts, or severe crisis, respond:
    "I hear you're in distress. I'm not qualified for crisis support. Please reach out to immediate professional help:
    Crisis Helpline: 988 (US)
    Emergency: 911 (US)
    Or contact a mental health professional directly."
    Approach

    Understand before suggesting
    Help break down overwhelming feelings
    Encourage self-reflection and mindfulness
    Celebrate small progress
    Remind users you're not a substitute for professional care
    """
    
    # Extract user messages from history
    user_messages = [msg["content"] for msg in history if msg["role"] == "user"]
    last_3_user = user_messages[-3:] if len(user_messages) >= 3 else user_messages
    
    if not last_3_user:
        # Return all expected values when no messages exist
        return base_prompt, {}, {}, base_prompt
    
    # Analyze tone using VADER sentiment analysis
    sentiment_scores = [analyzer.polarity_scores(msg) for msg in last_3_user]
    avg_compound = sum(s["compound"] for s in sentiment_scores) / len(sentiment_scores)
    
    # Build dynamic prompt
    dynamic_prompt = base_prompt
    
    # Add tone-specific guidance
    if avg_compound < -0.5:  # Distressed
        tone = "distressed"
        dynamic_prompt += """
        Current State: User appears to be in significant distress
        
        Priority Guidelines:
        - Acknowledge their pain immediately and validate their feelings
        - Use more supportive and validating statements
        - Keep responses gentle and reassuring
        - Focus on immediate emotional support
        - Be extra careful with suggestions or advice
        - Check if they have support systems in place
        """
    elif avg_compound < -0.2:  # Concerned
        tone = "concerned"
        dynamic_prompt += """
        Current State: User seems worried or troubled
        
        Approach Guidelines:
        - Validate their concerns explicitly
        - Use gentle exploration questions
        - Offer grounding techniques if appropriate
        - Help identify and break down specific worries
        - Maintain a calm and supportive presence
        """
    elif avg_compound < 0.1:  # Neutral
        tone = "neutral"
        dynamic_prompt += """
        Current State: User is in a neutral state
        
        Approach Guidelines:
        - Maintain engaging conversation
        - Explore their thoughts and feelings
        - Look for opportunities to deepen understanding
        """
    else:  # Positive
        tone = "positive" if avg_compound < 0.5 else "very_positive"
        dynamic_prompt += """
        Current State: User is in a positive state
        
        Approach Guidelines:
        - Reinforce positive elements
        - Explore what's working well
        - Help identify strategies to maintain progress
        """
    
    # Store for highlighting changes
    if 'last_prompt' not in st.session_state:
        st.session_state.last_prompt = dynamic_prompt
    
    # Compare with last prompt and highlight changes
    old_lines = st.session_state.last_prompt.split('\n')
    new_lines = dynamic_prompt.split('\n')
    
    highlighted_prompt = []
    for i, line in enumerate(new_lines):
        if i >= len(old_lines) or line.strip() != old_lines[i].strip():
            highlighted_prompt.append(f'<div class="prompt-change">{line}</div>')
        else:
            highlighted_prompt.append(line)
    
    # Update the stored prompt
    st.session_state.last_prompt = dynamic_prompt
    
    # Prepare metrics for display
    sentiment_metrics = {
        "Compound Score": round(avg_compound, 3),
        "Overall Tone": tone.capitalize()
    }
    
    engagement_metrics = {
        "Avg Word Count": round(sum(len(msg.split()) for msg in last_3_user) / len(last_3_user), 1),
        "Engagement Level": "Engaged" if len(last_3_user[-1].split()) >= 3 else "Disengaged"
    }
    
    return dynamic_prompt, sentiment_metrics, engagement_metrics, '\n'.join(highlighted_prompt)

def check_response_repetition(history, new_response):
    """Check if AI responses are becoming repetitive without progression"""
    # Get last 4 AI messages
    ai_messages = [msg["content"] for msg in history[-8:] if msg["role"] == "assistant"][-4:]
    if len(ai_messages) < 2:  # Need at least 2 messages to check repetition
        return False, ""
    
    # Calculate similarity ratios between consecutive messages
    similarities = []
    for i in range(len(ai_messages)-1):
        # Split messages into sentences for more granular comparison
        prev_sentences = set(ai_messages[i].split('. '))
        next_sentences = set(ai_messages[i+1].split('. '))
        
        # Calculate unique content ratio
        total_sentences = len(prev_sentences.union(next_sentences))
        common_sentences = len(prev_sentences.intersection(next_sentences))
        similarity = common_sentences / total_sentences if total_sentences > 0 else 0
        similarities.append(similarity)
    
    # Calculate similarity of new response with last response
    last_sentences = set(ai_messages[-1].split('. '))
    new_sentences = set(new_response.split('. '))
    total_sentences = len(last_sentences.union(new_sentences))
    common_sentences = len(last_sentences.intersection(new_sentences))
    new_similarity = common_sentences / total_sentences if total_sentences > 0 else 0
    similarities.append(new_similarity)
    
    # Check if recent responses are too similar
    avg_similarity = sum(similarities) / len(similarities)
    is_repetitive = avg_similarity > 0.6  # Lowered threshold and using sentence-based comparison
    
    adjustment = """
    The conversation seems to be repeating similar themes. Please:
    1. Acknowledge any new information or nuances in the user's messages
    2. Introduce a different perspective or approach
    3. Ask about a specific aspect that hasn't been explored
    4. Share a relevant coping technique or practical suggestion
    5. If appropriate, gently redirect the conversation to related but unexplored areas
    
    Remember to maintain empathy while helping the conversation progress naturally.
    
    Previous response themes to avoid: {themes}
    """.format(themes=", ".join(last_sentences)[:200] + "...")  # Include recent themes to avoid
    
    return is_repetitive, adjustment

def analyze_conversation(history):
    """Analyze conversation for various metrics"""
    if not history:
        return {}
    
    # Get messages by role
    user_messages = [msg["content"] for msg in history if msg["role"] == "user"]
    ai_messages = [msg["content"] for msg in history if msg["role"] == "assistant"]
    
    metrics = {
        "Conversation Stats": {
            "Total Messages": len(history),
            "User Messages": len(user_messages),
            "AI Responses": len(ai_messages),
            "Average User Message Length": round(sum(len(msg.split()) for msg in user_messages) / len(user_messages) if user_messages else 0, 1),
        },
        "Engagement Analysis": {
            "Short Responses (<5 words)": sum(1 for msg in user_messages if len(msg.split()) < 5),
            "Detailed Responses (>20 words)": sum(1 for msg in user_messages if len(msg.split()) > 20),
            "Questions Asked": sum(1 for msg in user_messages if "?" in msg),
        },
        "Sentiment Trends": {
            "Current Mood": None,
            "Mood Trend": None,
            "Emotional Range": None,
        },
        "Conversation Flow": {
            "Response Time Pattern": "Consistent" if ai_messages else "N/A",
            "Interaction Pattern": "Interactive" if len(history) > 4 else "Starting",
        }
    }
    
    # Analyze sentiment trends
    if user_messages:
        recent_sentiments = [analyzer.polarity_scores(msg)["compound"] for msg in user_messages[-3:]]
        current_sentiment = recent_sentiments[-1] if recent_sentiments else 0
        
        # Determine current mood
        if current_sentiment > 0.5:
            mood = "Very Positive"
        elif current_sentiment > 0.1:
            mood = "Positive"
        elif current_sentiment > -0.1:
            mood = "Neutral"
        elif current_sentiment > -0.5:
            mood = "Concerned"
        else:
            mood = "Distressed"
            
        # Analyze trend
        if len(recent_sentiments) > 1:
            trend = "Improving" if recent_sentiments[-1] > recent_sentiments[0] else "Declining" if recent_sentiments[-1] < recent_sentiments[0] else "Stable"
        else:
            trend = "Initial"
            
        # Calculate emotional range
        if len(recent_sentiments) > 1:
            emotional_range = max(recent_sentiments) - min(recent_sentiments)
            range_desc = "Wide" if emotional_range > 0.5 else "Moderate" if emotional_range > 0.2 else "Narrow"
        else:
            range_desc = "Initial"
            
        metrics["Sentiment Trends"].update({
            "Current Mood": mood,
            "Mood Trend": trend,
            "Emotional Range": range_desc
        })
    
    return metrics

def update_metrics():
    """Update the metrics display in the sidebar"""
    st.session_state.update_counter += 1
    counter = st.session_state.update_counter
    
    with st.session_state.metrics_container.container():
        st.title("Conversation Analytics")
        
        # Get analytics
        metrics = analyze_conversation(st.session_state.history)
        current_prompt, sentiment_metrics, engagement_metrics, highlighted_prompt = generate_dynamic_prompt(st.session_state.history)
        
        # Add CSS for metrics and prompts
        st.markdown("""
        <style>
        .metric-container {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .metric-label {
            font-weight: 600;
            color: var(--text-color);
        }
        .metric-value {
            color: var(--text-color);
            opacity: 0.9;
        }
        .prompt-change {
            background-color: rgba(255, 255, 0, 0.2);
            padding: 2px 4px;
            border-radius: 3px;
        }
        [data-testid="stMarkdownContainer"] {
            color: var(--text-color);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 1. Always display system configuration first
        st.markdown("### System Configuration")
        with st.expander("View Current Prompt", expanded=True):  # Set to auto-expand
            st.markdown(f"""
            <div class="prompt-container">
                {highlighted_prompt}
            </div>
            """, unsafe_allow_html=True)
            
            st.text_area(
                "Raw System Prompt",
                current_prompt,
                height=150,
                disabled=True,
                key=f"system_prompt_display_{counter}"
            )
        
        # Only display metrics if we have conversation history
        if metrics:
            # 2. Display Sentiment Trends
            if "Sentiment Trends" in metrics:
                st.markdown("### Sentiment Trends")
                for metric_name, value in metrics["Sentiment Trends"].items():
                    if value is not None:  # Only display non-None values
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">{metric_name}</div>
                            <div class="metric-value">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # 3. Display Detailed Sentiment
            if sentiment_metrics:
                st.markdown("### Detailed Sentiment")
                for i, (key, value) in enumerate(sentiment_metrics.items()):
                    if key == "Overall Tone":
                        color = {
                            "Very Positive": "#28a745",
                            "Positive": "#5cb85c",
                            "Neutral": "#6c757d",
                            "Concerned": "#ffc107",
                            "Distressed": "#dc3545"
                        }.get(value, "var(--text-color)")
                        
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">{key}</div>
                            <div class="metric-value" style="color: {color}">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">{key}</div>
                            <div class="metric-value">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # 4. Display remaining metrics sections
            for section_name, section_metrics in metrics.items():
                if section_name != "Sentiment Trends":  # Skip sentiment trends as already displayed
                    st.markdown(f"### {section_name}")
                    for metric_name, value in section_metrics.items():
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">{metric_name}</div>
                            <div class="metric-value">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)

def main():
    st.title("Asha")
    st.markdown("""
    > **Important**: This AI companion provides emotional support but is NOT a replacement for professional mental health care. 
    > If you're in crisis, please call 988 (US) or your local emergency services immediately.
    """)
    initialize_session_state()
    
    # Create persistent containers for metrics
    if 'metrics_container' not in st.session_state:
        st.session_state.metrics_container = st.sidebar.empty()
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
    
    # Initial metrics display
    update_metrics()
    
    # Display chat messages
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": user_input})
        
        # Update metrics immediately
        update_metrics()
        
        # Generate dynamic prompt for AI
        dynamic_prompt, _, _, _ = generate_dynamic_prompt(st.session_state.history)
        
        try:
            max_retries = 2
            retry_count = 0
            
            while retry_count < max_retries:
                with st.spinner("Thinking..."):
                    # Format conversation history for LangChain
                    messages = []
                    # Add system prompt
                    messages.append({"role": "system", "content": dynamic_prompt})
                    
                    # Add conversation history
                    for msg in st.session_state.history:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    # Generate response
                    response = st.session_state.model.invoke(messages)
                    ai_response = response.content
                    
                    # Check for repetition
                    is_repetitive, adjustment = check_response_repetition(st.session_state.history, ai_response)
                    
                    if not is_repetitive:
                        break
                    
                    # If repetitive, add adjustment and try again
                    messages.append({"role": "system", "content": adjustment})
                    retry_count += 1
            
            # Add final response to history
            st.session_state.history.append({"role": "assistant", "content": ai_response})
            
            # Display AI response
            with st.chat_message("assistant"):
                st.write(ai_response)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            with st.chat_message("assistant"):
                st.write("Sorry, I'm having trouble right now. Please try again later.")

if __name__ == "__main__":
    main()
import json
import hashlib
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from flask import Flask, render_template_string, request, jsonify
import re
from abc import ABC, abstractmethod

TAPEAGENTS_AVAILABLE = False
try:
    sys.path.append(os.path.abspath('tapeagents'))
    from tapeagents.llms import OpenrouterLLM, LLMOutput
    from tapeagents.agent import Agent
    from tapeagents.core import Prompt, PartialStep
    from tapeagents.dialog_tape import DialogTape, UserStep, AssistantStep, SystemStep
    from tapeagents.orchestrator import main_loop
    from tapeagents.environment import ToolCollectionEnvironment
    from tapeagents.tools.calculator import Calculator
    from tapeagents.nodes import StandardNode, Stop
    TAPEAGENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TapeAgents not available: {e}")

class LLMInterface(ABC):
    
    @abstractmethod
    def analyze_application(self, application_data: Dict[str, Any]) -> Dict[str, str]:
        pass

class SummaryAgentLLMProvider(LLMInterface):
    
    def __init__(self):
        if not TAPEAGENTS_AVAILABLE:
            raise ImportError("TapeAgents library is not available")
            
        api_key = "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"
        self.llm = OpenrouterLLM(
            model_name="meta-llama/llama-3.3-70b-instruct:free",
            api_token=api_key,
            parameters={"temperature": 0.1},
        )
        
        self.environment = ToolCollectionEnvironment(tools=[Calculator()])
        self.environment.initialize()
        
        try:
            with open("Synthetic Childcare Subsidy Regulation.md", "r") as f:
                regulation = f.read()
        except FileNotFoundError:
            regulation = "No regulation file found"
        
        system_prompt = f"""You are a childcare subsidy decision summary agent. Your job is to create clear, professional summaries of subsidy decisions that have already been made by a previous decision-making agent.

## CHILDCARE SUBSIDY REGULATION:
{regulation}

You will receive:
- Application data
- Eligibility assessment (the final decision)
- Validation flags (issues or criteria that were flagged)

## YOUR TASK FOR EACH CASE:
Create a professional summary of this childcare subsidy decision including:
1. **Case Overview**: Brief description of applicant and case
2. **Final Decision**: State the eligibility assessment clearly
3. **Validation Flags Explanation**: Explain what each validation flag means in plain language
4. **Key Factors**: Main factors that influenced the decision based on the flags
5. **Financial Impact**: If eligible, mention subsidy calculations
6. **Administrative Notes**: Important procedural information

## IMPORTANT GUIDELINES:
1. Summarizing the eligibility assessment clearly
2. Explaining validation flags in plain language and how they influenced the decision
3. Do NOT re-evaluate or change the eligibility assessment - only summarize what was already decided."""
        
        self.agent = Agent.create(
            llms=self.llm,
            nodes=[
                StandardNode(
                    name="summary",
                    system_prompt=system_prompt,
                    guidance="Review the decision data and flags provided, then create a professional summary of the childcare subsidy decision. Focus on clarity and completeness for official documentation.",
                ),
                Stop(),
            ],
            known_actions=self.environment.actions(),
            tools_description=self.environment.tools_description(),
        )
    
    def analyze_application(self, application_data: Dict[str, Any]) -> Dict[str, str]:
        flags_dict = asdict(application_data['validation_flags'])
        active_flags = [k for k, v in flags_dict.items() if v]
        
        if len(active_flags) > 5:
            decision = "REVIEW"
            analysis = f"Application requires human review due to multiple validation issues: {', '.join(active_flags[:3])}..."
        elif any(flag in ['income_threshold_exceeded', 'missing_required_fields'] for flag in active_flags):
            decision = "REVIEW"
            analysis = "Application has critical issues that require careful review."
        else:
            decision = "REVIEW"
            analysis = "Standard application processing required."
        
        app_id = application_data.get('application_id', 'Unknown')
        eligibility_assessment = application_data.get('eligibility_assessment', decision)
        
        if eligibility_assessment == 'REQUIRES_REVIEW' and app_id in decision_summaries:
            professional_summary = decision_summaries[app_id]
            print(f"📋 Using pre-generated summary for {app_id}")
        else:
            try:
                user_message = f"""## APPLICATION DATA:
{json.dumps(application_data, indent=2)}

## ELIGIBILITY DECISION:
{eligibility_assessment}

## VALIDATION FLAGS:
{", ".join(active_flags) if active_flags else "No validation flags"}"""
                
                tape = DialogTape(steps=[UserStep(content=user_message)])
                
                for event in main_loop(self.agent, tape, self.environment):
                    if event.agent_tape:
                        tape = event.agent_tape
                
                if len(tape.steps) >= 2 and hasattr(tape.steps[-2], 'reasoning'):
                    professional_summary = tape.steps[-2].reasoning
                elif len(tape.steps) >= 1 and hasattr(tape.steps[-1], 'reasoning'):
                    professional_summary = tape.steps[-1].reasoning
                else:
                    professional_summary = str(tape.steps[-1]) if tape.steps else analysis
                    
            except Exception as e:
                print(f"Summary agent error: {e}")
                professional_summary = analysis
        
        return {
            'analysis': professional_summary,
            'decision': decision,
            'reasoning': f"Based on validation flags: {', '.join(active_flags) if active_flags else 'No major issues detected'}"
        }

class MockLLMProvider(LLMInterface):
    
    def analyze_application(self, application_data: Dict[str, Any]) -> Dict[str, str]:
        flags_dict = asdict(application_data['validation_flags'])
        active_flags = [k for k, v in flags_dict.items() if v]
        
        app_id = application_data.get('application_id', 'Unknown')
        eligibility = application_data.get('eligibility_assessment', 'REVIEW')

        if len(active_flags) > 5:
            decision = "REVIEW"
            analysis = f"Application requires human review due to multiple validation issues: {', '.join(active_flags[:3])}..."
        elif any(flag in ['income_threshold_exceeded', 'missing_required_fields'] for flag in active_flags):
            decision = "REVIEW"
            analysis = "Application has critical issues that require careful review."
        else:
            decision = "REVIEW"
            analysis = "Standard application processing required."
        
        if eligibility == 'REQUIRES_REVIEW' and app_id in decision_summaries:
            professional_summary = decision_summaries[app_id]
            print(f"📋 Using pre-generated summary for {app_id}")
        else:
            income = application_data.get('household_income', 0)
            num_children = application_data.get('num_children', 0)
            hours = application_data.get('childcare_hours_requested', 0)
            
            professional_summary = f"""**CHILDCARE SUBSIDY DECISION SUMMARY**

**Case Overview:**
Application {app_id} - Household with {num_children} child{'ren' if num_children != 1 else ''}, requesting {hours} hours/month of childcare assistance.
Household income: ${income:,}

**Final Decision:** {eligibility}

**Validation Flags Analysis:**
{self._format_flag_explanations(active_flags) if active_flags else "No validation issues identified."}

**Key Factors:**
- Application assessed using standard eligibility criteria
- {len(active_flags)} validation flag(s) identified requiring attention
- Decision based on regulatory compliance and risk assessment

**Administrative Notes:**
This summary was generated using automated decision support tools. Manual review recommended for complex cases."""
        
        return {
            'analysis': professional_summary,
            'decision': decision,
            'reasoning': f"Based on validation flags: {', '.join(active_flags) if active_flags else 'No major issues detected'}"
        }
    
    def _format_flag_explanations(self, active_flags: List[str]) -> str:
        explanations = {
            'income_threshold_exceeded': '• Income exceeds 75th percentile threshold - may affect subsidy eligibility',
            'missing_required_fields': '• Required application fields are incomplete - documentation needed',
            'employment_status_invalid': '• Employment status does not match accepted categories',
            'child_age_inconsistency': '• Child age information is inconsistent or exceeds program limits',
            'high_hours_request': '• Requested childcare hours exceed standard thresholds',
            'documentation_incomplete': '• Supporting documentation is missing or insufficient',
            'inconsistent_data_format': '• Data format inconsistencies detected in application'
        }
        
        result = []
        for flag in active_flags:
            if flag in explanations:
                result.append(explanations[flag])
            else:
                result.append(f"• {flag.replace('_', ' ').title()}: Requires review")
        
        return '\n'.join(result)

if TAPEAGENTS_AVAILABLE:
    class ChildcareSupportAgent(Agent[DialogTape]):
        
        def make_prompt(self, tape: DialogTape):
            messages = []
            for step in tape.steps:
                if isinstance(step, SystemStep):
                    messages.append({"role": "system", "content": step.content})
                elif isinstance(step, UserStep):
                    messages.append({"role": "user", "content": step.content})
                elif isinstance(step, AssistantStep):
                    messages.append({"role": "assistant", "content": step.content})
            return Prompt(messages=messages)
            
        def generate_steps(self, tape: DialogTape, llm_stream):
            buffer = []
            for event in llm_stream:
                if event.chunk:
                    buffer.append(event.chunk)
                    yield PartialStep(step=AssistantStep(content="".join(buffer)))
                elif event.output:
                    final_content = event.output.content or "".join(buffer)
                    yield AssistantStep(content=final_content)
                    return
                else:
                    raise ValueError(f"Unknown event type from LLM: {event}")

        def make_llm_output(self, tape: DialogTape, index: int):
            step = tape.steps[index]
            if not isinstance(step, AssistantStep):
                raise ValueError("Expected AssistantStep")
            return LLMOutput(content=step.content)

class ChatbotInterface:
    
    def __init__(self, llm_provider: LLMInterface):
        self.llm_provider = llm_provider
        self.enhanced_agent = None
        self.conversation_tape = None
        
        print(f"🔍 TAPEAGENTS_AVAILABLE: {TAPEAGENTS_AVAILABLE}")
        
        # Try the simplest possible approach - direct LLM usage
        try:
            print("🔧 Attempting simple LLM initialization...")
            self._initialize_simple_llm()
            print(f"🔍 After simple init - enhanced_agent: {self.enhanced_agent is not None}")
        except Exception as e:
            print(f"⚠️  Simple LLM initialization failed: {e}")
            import traceback
            traceback.print_exc()
            
            # If simple init fails, try the complex method
            try:
                print("🔧 Trying complex initialization method...")
                self._initialize_enhanced_chatbot_force()
                print(f"🔍 After complex init - enhanced_agent: {self.enhanced_agent is not None}")
            except Exception as e2:
                print(f"⚠️  Complex initialization also failed: {e2}")
                
                # Last resort - try with TAPEAGENTS_AVAILABLE flag
                if TAPEAGENTS_AVAILABLE:
                    try:
                        print("🔧 Trying original initialization method...")
                        self._initialize_enhanced_chatbot()
                        print(f"🔍 After original init - enhanced_agent: {self.enhanced_agent is not None}")
                    except Exception as e3:
                        print(f"⚠️  All initialization methods failed: {e3}")
                else:
                    print("ℹ️  TapeAgents not available globally, all initialization failed")
    
    def _initialize_enhanced_chatbot(self):
        if not TAPEAGENTS_AVAILABLE:
            return
            
        api_key = "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"
        llm = OpenrouterLLM(
            model_name="meta-llama/llama-3.3-70b-instruct:free",
            api_token=api_key,
            parameters={"temperature": 0.1},
        )
        
        try:
            self.enhanced_agent = ChildcareSupportAgent.create(llm, name="childcare_agent")
            print("✅ ChildcareSupportAgent created successfully")
            
            # Test the LLM directly
            print("🧪 Testing LLM directly...")
            test_prompt = Prompt(messages=[{"role": "user", "content": "Hello, can you help me with childcare applications?"}])
            test_stream = llm.stream(test_prompt)
            test_response = ""
            for event in test_stream:
                if event.output:
                    test_response = event.output.content
                    break
            print(f"🧪 Direct LLM test response: {test_response[:100]}...")
            
        except Exception as e:
            print(f"❌ Failed to create ChildcareSupportAgent: {e}")
            import traceback
            traceback.print_exc()
            self.enhanced_agent = None
        
        try:
            with open("Synthetic Childcare Subsidy Regulation.md", "r") as f:
                regulation = f.read()
        except FileNotFoundError:
            regulation = "No regulation file found"
        
        try:
            with open("summaryoutput.txt", "r") as f:
                summary = f.read()
        except FileNotFoundError:
            summary = "No summary file found"
        
        flag_guide = """
Childcare Application Flags:
- income_threshold_exceeded: Income exceeds 75th percentile threshold
- missing_required_fields: Required application fields are incomplete
- employment_status_invalid: Employment status does not match accepted categories
- child_age_inconsistency: Child age information is inconsistent or exceeds program limits
- high_hours_request: Requested childcare hours exceed standard thresholds
- documentation_incomplete: Supporting documentation is missing or insufficient
- inconsistent_data_format: Data format inconsistencies detected in application

Use this guide to explain flags to municipality workers.
"""
        
        system_content = (
            "You are a helpful childcare support assistant for municipality workers reviewing childcare subsidy applications. "
            "Your role is to explain application flags, decisions, and regulations in a clear and helpful manner. "
            "Always provide detailed, informative responses to help municipality workers understand the childcare subsidy system. "
            "When asked about flags, explain what they mean and their implications. "
            "When asked general questions, provide helpful information about the childcare subsidy process. "
            + flag_guide +
            f"\n\nRegulations: {regulation[:1000]}..." +
            f"\n\nDecision summaries available: {summary[:500]}..."
        )
        
        self.conversation_tape = DialogTape(
            context=None,
            steps=[SystemStep(content=system_content)]
        )
        print("✅ Enhanced chatbot conversation tape initialized")
    
    def _initialize_simple_llm(self):
        # Try the simplest possible approach using requests
        try:
            import requests
            import json
            
            print("🔍 Testing direct API call to OpenRouter...")
            
            api_key = "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"
            
            # Test direct API call
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, respond with 'Direct API working!'"}
                ],
                "temperature": 0.1,
                "max_tokens": 100
            }
            
            print(f"🔍 Making API call to {url}")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            print(f"🔍 API response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                test_response = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"🧪 Direct API test response: {test_response}")
                
                if test_response:
                    # Create a simple agent using direct API calls
                    class DirectAPIAgent:
                        def __init__(self, api_key):
                            self.api_key = api_key
                            self.url = "https://openrouter.ai/api/v1/chat/completions"
                            self.headers = {
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json"
                            }
                        
                        def respond(self, user_message, system_context="You are a helpful assistant."):
                            data = {
                                "model": "meta-llama/llama-3.3-70b-instruct:free",
                                "messages": [
                                    {"role": "system", "content": system_context},
                                    {"role": "user", "content": user_message}
                                ],
                                "temperature": 0.1,
                                "max_tokens": 500
                            }
                            
                            try:
                                response = requests.post(self.url, headers=self.headers, json=data, timeout=30)
                                if response.status_code == 200:
                                    result = response.json()
                                    return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                                else:
                                    print(f"❌ API error: {response.status_code} - {response.text}")
                                    return f"API Error: {response.status_code}"
                            except Exception as e:
                                print(f"❌ Request error: {e}")
                                return f"Request Error: {e}"
                    
                    self.enhanced_agent = DirectAPIAgent(api_key)
                    print("✅ Created DirectAPIAgent")
                    
                    # Test the agent
                    test_response = self.enhanced_agent.respond(
                        "What are childcare application flags?",
                        "You are a childcare support assistant. Explain application flags clearly."
                    )
                    print(f"🧪 DirectAPIAgent test response: {test_response[:100]}...")
                    
                    if test_response and len(test_response) > 10 and "Error" not in test_response:
                        self.system_context = (
                            "You are a helpful childcare support assistant for municipality workers. "
                            "Explain application flags, decisions, and regulations clearly."
                        )
                        print("✅ Simple LLM initialization completed successfully with DirectAPIAgent")
                        return
                    else:
                        raise Exception("DirectAPIAgent test failed or returned error")
                else:
                    raise Exception("No response from direct API test")
            else:
                print(f"❌ API call failed: {response.status_code} - {response.text}")
                raise Exception(f"API call failed with status {response.status_code}")
        
        except Exception as e:
            print(f"❌ Simple LLM initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _initialize_enhanced_chatbot_force(self):
        # Try to import tapeagents directly and initialize
        try:
            import sys
            import os
            
            # Try different possible paths for tapeagents
            possible_paths = [
                os.path.abspath('tapeagents'),
                os.path.abspath('../tapeagents'),
                os.path.abspath('./tapeagents'),
                '/usr/local/lib/python3.*/site-packages',
                '/opt/conda/lib/python3.*/site-packages'
            ]
            
            for path in possible_paths:
                if path not in sys.path:
                    sys.path.append(path)
                    print(f"🔍 Added to path: {path}")
            
            print("🔍 Current Python path:")
            for p in sys.path[-5:]:
                print(f"  - {p}")
            
            # Try importing step by step
            print("🔍 Trying to import tapeagents modules...")
            from tapeagents.llms import OpenrouterLLM, LLMOutput
            print("✅ Imported OpenrouterLLM, LLMOutput")
            
            from tapeagents.agent import Agent
            print("✅ Imported Agent")
            
            from tapeagents.core import Prompt, PartialStep
            print("✅ Imported Prompt, PartialStep")
            
            from tapeagents.dialog_tape import DialogTape, UserStep, AssistantStep, SystemStep
            print("✅ Imported DialogTape classes")
            
            # Create LLM and test it first
            api_key = "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"
            print(f"🔍 Using API key: {api_key[:20]}...")
            
            llm = OpenrouterLLM(
                model_name="meta-llama/llama-3.3-70b-instruct:free",
                api_token=api_key,
                parameters={"temperature": 0.1},
            )
            print("✅ Created OpenrouterLLM instance")
            
            # Test the LLM directly first
            print("🧪 Testing LLM directly before creating agent...")
            test_prompt = Prompt(messages=[{"role": "user", "content": "Hello, respond with 'LLM is working!'"}])
            test_stream = llm.stream(test_prompt)
            test_response = ""
            event_count = 0
            for event in test_stream:
                event_count += 1
                print(f"🔍 LLM event {event_count}: {type(event)}")
                if hasattr(event, 'chunk') and event.chunk:
                    test_response += event.chunk
                    print(f"🔍 Chunk: {event.chunk}")
                elif hasattr(event, 'output') and event.output:
                    test_response = event.output.content
                    print(f"🔍 Final output: {event.output.content}")
                    break
            print(f"🧪 Direct LLM test response: {test_response}")
            
            if not test_response:
                raise Exception("LLM test failed - no response received")
            
            # Store the working LLM for direct use
            self.direct_llm = llm
            print("✅ Stored direct LLM instance")
            
            # Try to create a simple agent
            print("🔍 Creating minimal agent...")
            
            class SimpleChatAgent:
                def __init__(self, llm):
                    self.llm = llm
                
                def respond(self, user_message, system_context="You are a helpful assistant."):
                    messages = [
                        {"role": "system", "content": system_context},
                        {"role": "user", "content": user_message}
                    ]
                    prompt = Prompt(messages=messages)
                    stream = self.llm.stream(prompt)
                    
                    response = ""
                    for event in stream:
                        if hasattr(event, 'chunk') and event.chunk:
                            response += event.chunk
                        elif hasattr(event, 'output') and event.output:
                            response = event.output.content
                            break
                    return response
            
            self.simple_agent = SimpleChatAgent(llm)
            print("✅ Created simple chat agent")
            
            # Test the simple agent
            test_response = self.simple_agent.respond(
                "What are childcare application flags?",
                "You are a childcare support assistant. Explain application flags clearly."
            )
            print(f"🧪 Simple agent test response: {test_response[:100]}...")
            
            if test_response and len(test_response) > 10:
                print("✅ Simple agent is working! Using this instead of complex agent.")
                self.enhanced_agent = self.simple_agent
                
                # Create a simple conversation state
                self.conversation_history = []
                system_content = (
                    "You are a helpful childcare support assistant for municipality workers. "
                    "Explain application flags, decisions, and regulations clearly."
                )
                self.system_context = system_content
                
                print("✅ Forced initialization with simple agent completed successfully")
                return
            
            # If we get here, the simple agent didn't work, so the original error stands
            raise Exception("Simple agent test failed")
            
        except Exception as e:
            print(f"❌ Forced initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _reset_conversation_if_needed(self):
        if self.conversation_tape and len(self.conversation_tape.steps) > 20:
            print("🔄 Resetting conversation tape (too many steps)")
            system_step = self.conversation_tape.steps[0]
            self.conversation_tape = DialogTape(
                context=None,
                steps=[system_step]
            )
    
    def get_application_summary(self, application_id: str) -> str:
        app = None
        for processed_app in processed_applications:
            if processed_app.application_id == application_id:
                app = processed_app
                break
        
        if not app:
            return "Application not found."
        
        flags_dict = asdict(app.validation_flags)
        active_flags = [k.replace('_', ' ').title() for k, v in flags_dict.items() if v]
        
        summary = f"""Application {app.application_id} Summary:
- Household Income: ${app.household_income:,}
- Children: {app.num_children} (ages: {', '.join(map(str, app.child_ages)) if app.child_ages else 'N/A'})
- Employment: {app.employment_status.title()}
- Housing: {app.housing_situation.replace('_', ' ').title()}
- Hours Requested: {app.childcare_hours_requested}/month
- Active Flags: {', '.join(active_flags) if active_flags else 'None'}
- LLM Analysis: {app.llm_analysis or 'Not yet analyzed'}
- Decision: {app.llm_decision or 'Pending'}"""
        return summary
    
    def process_user_query(self, query: str, application_id: str = None) -> str:
        print(f"🔍 Debug: enhanced_agent exists: {self.enhanced_agent is not None}")
        print(f"🔍 Debug: enhanced_agent type: {type(self.enhanced_agent)}")
        
        if self.enhanced_agent:
            try:
                print(f"🤖 Processing query with enhanced agent: {query[:50]}...")
                
                # Check if it's our simple agent or complex agent
                if hasattr(self.enhanced_agent, 'respond'):
                    # Simple agent
                    print("🔍 Using simple agent...")
                    
                    context = ""
                    if application_id:
                        app_summary = self.get_application_summary(application_id)
                        context = f"Context for application {application_id}: {app_summary}\n\n"
                    
                    full_query = context + query
                    system_context = getattr(self, 'system_context', 
                        "You are a helpful childcare support assistant for municipality workers. "
                        "Explain application flags, decisions, and regulations clearly."
                    )
                    
                    print(f"🔍 Calling simple agent with query: {full_query[:100]}...")
                    response_content = self.enhanced_agent.respond(full_query, system_context)
                    print(f"✅ Simple agent response: {len(response_content)} chars")
                    print(f"🔍 Response preview: {response_content[:100]}...")
                    
                    if response_content and response_content.strip():
                        return response_content
                    else:
                        print("⚠️  Empty response from simple agent, using fallback")
                        return self._fallback_query_processing(query, application_id)
                
                elif hasattr(self.enhanced_agent, 'run') and self.conversation_tape:
                    # Complex agent with conversation tape
                    print("🔍 Using complex agent with conversation tape...")
                    
                    self._reset_conversation_if_needed()
                    
                    context = ""
                    if application_id:
                        app_summary = self.get_application_summary(application_id)
                        context = f"Context for application {application_id}: {app_summary}\n\n"
                    
                    full_query = context + query
                    print(f"🔍 Full query to LLM: {full_query[:100]}...")
                    
                    self.conversation_tape = self.conversation_tape.append(UserStep(content=full_query))
                    print(f"🔍 Tape has {len(self.conversation_tape.steps)} steps")
                    
                    new_tape = None
                    response_content = ""
                    event_count = 0
                    
                    print("🔄 Starting LLM generation...")
                    for event in self.enhanced_agent.run(self.conversation_tape):
                        event_count += 1
                        print(f"🔍 Event {event_count}: {type(event).__name__}")
                        if event.partial_step:
                            response_content = event.partial_step.step.content
                            print(f"🔍 Partial response: {response_content[:50]}...")
                        if event.final_tape:
                            new_tape = event.final_tape
                            print("🔍 Got final tape")
                    
                    print(f"📊 Processed {event_count} events from LLM")
                    
                    if new_tape:
                        self.conversation_tape = new_tape
                        last_step = self.conversation_tape.steps[-1]
                        if hasattr(last_step, 'content'):
                            response_content = last_step.content
                            print(f"✅ Final response from enhanced agent: {len(response_content)} chars")
                            print(f"🔍 Response preview: {response_content[:100]}...")
                    
                    if response_content and response_content.strip():
                        return response_content
                    else:
                        print("⚠️  Empty response from enhanced agent, using fallback")
                        return self._fallback_query_processing(query, application_id)
                
                else:
                    print("⚠️  Enhanced agent exists but doesn't have expected methods")
                    return self._fallback_query_processing(query, application_id)
                
            except Exception as e:
                print(f"❌ Enhanced chatbot error: {e}")
                import traceback
                traceback.print_exc()
                return self._fallback_query_processing(query, application_id)
        else:
            print(f"ℹ️  Using fallback query processing (no enhanced agent)")
            return self._fallback_query_processing(query, application_id)
    
    def _fallback_query_processing(self, query: str, application_id: str = None) -> str:
        query_lower = query.lower()
        
        if application_id:
            if "summary" in query_lower or "overview" in query_lower:
                return self.get_application_summary(application_id)
            elif "flag" in query_lower or "issue" in query_lower:
                return self._get_flag_details(application_id)
            elif "decision" in query_lower or "recommend" in query_lower:
                return self._get_decision_reasoning(application_id)
            elif "explain" in query_lower:
                app_summary = self.get_application_summary(application_id)
                return f"{app_summary}\n\nThis application requires manual review by a municipality worker to determine final eligibility."
        
        if "total" in query_lower or "count" in query_lower:
            return f"Currently processing {len(processed_applications)} applications."
        elif "help" in query_lower:
            return """I can help you with:
- Application summaries: Ask for "summary of [APPLICATION_ID]"
- Flag explanations: Ask about "flags for [APPLICATION_ID]"
- Decision reasoning: Ask about "decision for [APPLICATION_ID]"
- General statistics: Ask about "total applications"
- Application analysis: Provide an application ID for specific details"""
        elif "regulation" in query_lower or "rule" in query_lower:
            return "Applications are evaluated based on the Childcare Subsidy Access Regulation (CSAR) - 2025, considering factors like household income, employment status, number of children, and childcare hours requested."
        elif "flag" in query_lower and not application_id:
            return """Common validation flags include:
- Income Threshold Exceeded: Household income exceeds 75th percentile
- Missing Required Fields: Incomplete application data
- Employment Status Invalid: Employment status doesn't match accepted categories
- Child Age Inconsistency: Child age information is inconsistent or exceeds limits
- High Hours Request: Requested childcare hours exceed standard thresholds
- Documentation Incomplete: Missing supporting documents"""
        
        return "I can help you analyze specific applications. Please provide an application ID or ask about overall statistics. Type 'help' for more options."
    
    def _get_flag_details(self, application_id: str) -> str:
        app = None
        for processed_app in processed_applications:
            if processed_app.application_id == application_id:
                app = processed_app
                break
        
        if not app:
            return "Application not found."
        
        flags_dict = asdict(app.validation_flags)
        active_flags = [(k.replace('_', ' ').title(), v) for k, v in flags_dict.items() if v]
        
        if not active_flags:
            return f"Application {application_id} has no validation flags."
        
        details = f"Application {application_id} validation flags:\n"
        for flag_name, _ in active_flags:
            details += f"- {flag_name}\n"
        
        return details
    
    def _get_decision_reasoning(self, application_id: str) -> str:
        app = None
        for processed_app in processed_applications:
            if processed_app.application_id == application_id:
                app = processed_app
                break
        
        if not app:
            return "Application not found."
        
        if app.llm_reasoning:
            return f"Decision reasoning for {application_id}: {app.llm_reasoning}"
        else:
            return f"Application {application_id} has not been analyzed by LLM yet."

@dataclass
class ValidationFlags:
    inconsistent_data_format: bool = False
    missing_required_fields: bool = False
    income_threshold_exceeded: bool = False
    employment_status_invalid: bool = False
    child_age_inconsistency: bool = False
    high_hours_request: bool = False
    documentation_incomplete: bool = False
    human_review_required: bool = False

@dataclass
class ProcessedApplication:
    application_id: str
    household_income: int
    employment_status: str
    num_children: int
    child_ages: List[int]
    childcare_hours_requested: int
    housing_situation: str
    partner_employed: bool
    validation_flags: ValidationFlags
    eligibility_assessment: str
    processed_timestamp: str
    reviewer_notes: Optional[str] = None
    reviewer_decision: Optional[str] = None
    review_timestamp: Optional[str] = None
    reviewer_reason: Optional[str] = None
    contact_email: Optional[str] = None
    llm_analysis: Optional[str] = None
    llm_decision: Optional[str] = None
    llm_reasoning: Optional[str] = None

class SubsidyApplicationProcessor:
    def __init__(self):
        self.INCOME_THRESHOLD_PERCENTILE = 75
        self.MAX_CHILD_AGE = 12
        self.STANDARD_WORK_HOURS_MONTHLY = 160
        self.HIGH_HOURS_THRESHOLD_MONTHLY  = 200
        
        self.VALID_EMPLOYMENT_STATUSES = {
            'employed', 'part-time', 'self-employed', 'freelancer', 
            'student', 'unemployed'
        }
        
        self.MUNICIPAL_MEDIAN_INCOME = 45000
        self.INCOME_THRESHOLDS = {
            '25th_percentile': self.MUNICIPAL_MEDIAN_INCOME * 0.65,
            '50th_percentile': self.MUNICIPAL_MEDIAN_INCOME * 1.0,
            '75th_percentile': self.MUNICIPAL_MEDIAN_INCOME * 1.35
        }
        
        self.next_id_counter = 1

    def anonymize_applicant(self, name: str) -> str:
        return hashlib.sha256(name.encode()).hexdigest()[:12]
    
    def _get_next_unique_id(self) -> str:
        max_id = 0
        
        for app in processed_applications:
            app_id = app.application_id
            if app_id.startswith('A') and len(app_id) == 4 and app_id[1:].isdigit():
                id_num = int(app_id[1:])
                max_id = max(max_id, id_num)
        
        try:
            with open('Merged Subsidy Applications.json', 'r') as f:
                original_data = json.load(f)
                for app in original_data:
                    app_id = app.get('application_id', '')
                    if app_id.startswith('A') and len(app_id) == 4 and app_id[1:].isdigit():
                        id_num = int(app_id[1:])
                        max_id = max(max_id, id_num)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        self.next_id_counter = max(max_id + 1, self.next_id_counter)
        next_id = f"A{self.next_id_counter:03d}"
        self.next_id_counter += 1
        return next_id

    def normalize_application_data(self, app_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        
        application_id = app_data.get('application_id')
        if not application_id or application_id.strip() == '':
            application_id = self._get_next_unique_id()
        normalized['application_id'] = application_id
        
        income = app_data.get('household_income', 0)
        try:
            normalized['household_income'] = int(income) if income is not None else 0
        except (ValueError, TypeError):
            normalized['household_income'] = 0
        
        employment_status = app_data.get('employment_status', 'unknown')
        normalized['employment_status'] = employment_status.lower().strip() if employment_status else 'unknown'
        

        num_children_field1 = app_data.get('num_children')
        num_children_field2 = app_data.get('number_of_children')
        child_ages = app_data.get('child_ages', [])
        
        if num_children_field1 is not None:
            try:
                normalized['num_children'] = int(num_children_field1)
            except (ValueError, TypeError):
                normalized['num_children'] = len(child_ages) if child_ages else 0
        elif num_children_field2 is not None:
            try:
                normalized['num_children'] = int(num_children_field2)
            except (ValueError, TypeError):
                normalized['num_children'] = len(child_ages) if child_ages else 0
        else:
            normalized['num_children'] = len(child_ages) if child_ages else 0
            
        normalized['child_ages'] = child_ages if isinstance(child_ages, list) else []
        

        hours_field1 = app_data.get('childcare_hours_requested')
        hours_field2 = app_data.get('requested_hours')
        hours = hours_field1 if hours_field1 is not None else hours_field2
        try:
            normalized['childcare_hours_requested'] = int(hours) if hours is not None else 0
        except (ValueError, TypeError):
            normalized['childcare_hours_requested'] = 0
        

        housing = app_data.get('housing_situation', 'unknown')
        if housing == 'rental':
            housing = 'rented'
        elif housing == 'municipal housing':
            housing = 'municipal_housing'
        normalized['housing_situation'] = housing
        

        normalized['partner_employed'] = bool(app_data.get('partner_employed', False))
        
        return normalized

    def validate_application(self, app_data: Dict[str, Any]) -> ValidationFlags:
        flags = ValidationFlags()
        
        required_fields = ['household_income', 'employment_status', 'num_children', 'childcare_hours_requested']
        if any(app_data.get(field) is None for field in required_fields):
            flags.missing_required_fields = True
            

        if app_data.get('household_income', 0) > self.INCOME_THRESHOLDS['75th_percentile']:
            flags.income_threshold_exceeded = True
            

        employment_status = app_data.get('employment_status', '').lower()
        if employment_status not in self.VALID_EMPLOYMENT_STATUSES:
            flags.employment_status_invalid = True
            

        num_children = app_data.get('num_children', 0)
        child_ages = app_data.get('child_ages', [])
        
        if num_children != len(child_ages) and len(child_ages) > 0:
            flags.child_age_inconsistency = True
            

        if any(age > self.MAX_CHILD_AGE for age in child_ages):
            flags.child_age_inconsistency = True
            

        hours_requested = app_data.get('childcare_hours_requested', 0)
        if hours_requested > self.HIGH_HOURS_THRESHOLD_MONTHLY:
            flags.high_hours_request = True
            

            

        original_flags = app_data.get('flags', {})
        if original_flags.get('incomplete_docs', False):
            flags.documentation_incomplete = True
            
        return flags

    def assess_eligibility(self, app_data: Dict[str, Any], flags: ValidationFlags) -> str:
        flags_dict = asdict(flags)
        critical_flags = [
            'missing_required_fields',
            'income_threshold_exceeded', 
            'employment_status_invalid',
            'child_age_inconsistency',
            'inconsistent_data_format',
            'documentation_incomplete'
        ]
        
        if any(flags_dict.get(flag, False) for flag in critical_flags):
            return "REQUIRES_REVIEW"
        
        household_income = app_data.get('household_income', 0)
        employment_status = app_data.get('employment_status', '').lower()
        num_children = app_data.get('num_children', 0)
        
        if (household_income > 0 and 
            employment_status in ['employed', 'part-time', 'self-employed', 'unemployed'] and
            num_children > 0):
            return "APPROVED"
        
        return "REQUIRES_REVIEW"


    def process_application(self, app_data: Dict[str, Any]) -> ProcessedApplication:
        normalized_data = self.normalize_application_data(app_data)
        validation_flags = self.validate_application(normalized_data)
        
        if (app_data.get('application_id') != normalized_data.get('application_id') or
            app_data.get('num_children') != normalized_data.get('num_children') or
            app_data.get('childcare_hours_requested') != normalized_data.get('childcare_hours_requested')):
            validation_flags.inconsistent_data_format = True
            
        eligibility = self.assess_eligibility(normalized_data, validation_flags)
        
        return ProcessedApplication(
            application_id=normalized_data['application_id'],
            household_income=normalized_data['household_income'],
            employment_status=normalized_data['employment_status'],
            num_children=normalized_data['num_children'],
            child_ages=normalized_data['child_ages'],
            childcare_hours_requested=normalized_data['childcare_hours_requested'],
            housing_situation=normalized_data['housing_situation'],
            partner_employed=normalized_data['partner_employed'],
            validation_flags=validation_flags,
            eligibility_assessment=eligibility,
            processed_timestamp=datetime.now().isoformat()
        )

    def process_all_applications(self, applications_data: List[Dict[str, Any]]) -> List[ProcessedApplication]:
        return [self.process_application(app) for app in applications_data]

    def load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {file_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading JSON file {file_path}: {str(e)}")


processed_applications = []
processor = SubsidyApplicationProcessor()

decision_summaries = {}
try:
    with open('decision_summaries.json', 'r', encoding='utf-8') as f:
        summaries_data = json.load(f)
        for summary in summaries_data:
            case_id = summary.get('case_id')
            if case_id:
                decision_summaries[case_id] = summary.get('summary', '')
    print(f"✅ Loaded {len(decision_summaries)} decision summaries")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"⚠️  Could not load decision summaries: {e}")
    decision_summaries = {}


if TAPEAGENTS_AVAILABLE:
    try:
        llm_provider = SummaryAgentLLMProvider()
        print("✅ Summary-agent LLM provider initialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize summary-agent, falling back to mock provider: {e}")
        llm_provider = MockLLMProvider()
else:
    print("ℹ️  TapeAgents not available, using MockLLMProvider with enhanced summaries")
    llm_provider = MockLLMProvider()

chatbot = ChatbotInterface(llm_provider)


app = Flask(__name__)
app.secret_key = 'subsidy_review_app_2025'

def markdown_to_html(text):
    if not text:
        return text
    
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'^#{1,6}\s+(.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^•\s+(.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^-\s+(.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    if '<li>' in text:
        text = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', text, flags=re.DOTALL)
        text = re.sub(r'</ul>\s*<ul>', '', text)
    
    text = text.replace('\n\n', '<br><br>')
    text = text.replace('\n', '<br>')
    
    return text

app.jinja_env.filters['markdown_to_html'] = markdown_to_html

def save_processing_results():
    output_file = "processed_applications.json"
    
    try:

        results = []
        for app in processed_applications:
            app_dict = asdict(app)
            results.append(app_dict)
        

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Processing results saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving processing results: {e}")
        return None

def load_and_process_applications():
    global processed_applications, processor
    
    json_file_path = "Merged Subsidy Applications.json"
    
    try:
        applications = processor.load_json_file(json_file_path)
        processed_applications = processor.process_all_applications(applications)
        print(f"Loaded and processed {len(processed_applications)} applications")
        
        for app in processed_applications:
            if app.eligibility_assessment == 'REQUIRES_REVIEW' and app.application_id in decision_summaries:
                app.llm_analysis = decision_summaries[app.application_id]
                app.llm_decision = "REVIEW"
                app.llm_reasoning = "Summary generated using AI decision support system"
                print(f"📋 Applied pre-generated summary to {app.application_id}")

        save_processing_results()
        
    except Exception as e:
        print(f"Error loading applications: {e}")
        processed_applications = []

@app.route('/')
def dashboard():
    pending_count = sum(1 for app in processed_applications 
                       if app.eligibility_assessment == 'REQUIRES_REVIEW' and not app.reviewer_decision)
    approved_count = sum(1 for app in processed_applications 
                        if app.eligibility_assessment == 'APPROVED' or app.reviewer_decision == 'APPROVED')
    rejected_count = sum(1 for app in processed_applications if app.reviewer_decision == 'REJECTED')
    
    return render_template_string(
        DASHBOARD_TEMPLATE, 
        applications=processed_applications,
        pending_count=pending_count,
        approved_count=approved_count,
        rejected_count=rejected_count
    )

@app.route('/application/<application_id>')
def application_detail(application_id):
    app_data = None
    for app in processed_applications:
        if app.application_id == application_id:
            app_data = app
            break
    
    if not app_data:
        return "Application not found", 404
    

    flags_dict = asdict(app_data.validation_flags)
    active_flags = [k.replace('_', ' ').title() for k, v in flags_dict.items() if v]
    
    return render_template_string(
        DETAIL_TEMPLATE, 
        application=app_data, 
        active_flags=active_flags,
        flags_dict=flags_dict
    )

@app.route('/api/applications')
def api_applications():
    return jsonify([asdict(app) for app in processed_applications])

@app.route('/api/application/<application_id>')
def api_application_detail(application_id):
    for app in processed_applications:
        if app.application_id == application_id:
            return jsonify(asdict(app))
    return jsonify({"error": "Application not found"}), 404

@app.route('/api/chatbot', methods=['POST'])
def chatbot_query():
    try:
        data = request.get_json()
        query = data.get('query', '')
        application_id = data.get('application_id', None)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        response = chatbot.process_user_query(query, application_id)
        return jsonify({"response": response})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chatbot/test', methods=['POST'])
def test_chatbot():
    try:
        data = request.get_json()
        test_query = data.get('query', 'Hello, can you help me understand application flags?')
        
        chatbot_status = {
            "tapeagents_available": TAPEAGENTS_AVAILABLE,
            "enhanced_agent_available": chatbot.enhanced_agent is not None,
            "conversation_tape_available": chatbot.conversation_tape is not None
        }
        
        if chatbot.enhanced_agent and chatbot.conversation_tape:
            response = chatbot.process_user_query(test_query)
            return jsonify({
                "status": chatbot_status,
                "test_query": test_query,
                "response": response,
                "agent_type": "enhanced"
            })
        else:
            response = chatbot._fallback_query_processing(test_query)
            return jsonify({
                "status": chatbot_status,
                "test_query": test_query,
                "response": response,
                "agent_type": "fallback"
            })
            
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": {
                "tapeagents_available": TAPEAGENTS_AVAILABLE,
                "enhanced_agent_available": chatbot.enhanced_agent is not None if 'chatbot' in globals() else False,
                "conversation_tape_available": chatbot.conversation_tape is not None if 'chatbot' in globals() else False
            }
        }), 500

@app.route('/api/chatbot/direct', methods=['POST'])
def test_direct_llm():
    print(f"🔍 Direct test called - enhanced_agent exists: {chatbot.enhanced_agent is not None}")
    print(f"🔍 Enhanced agent type: {type(chatbot.enhanced_agent) if chatbot.enhanced_agent else 'None'}")
    
    if not chatbot.enhanced_agent:
        error_response = {
            "error": "Enhanced chatbot not available", 
            "tapeagents_available": TAPEAGENTS_AVAILABLE,
            "enhanced_agent_exists": chatbot.enhanced_agent is not None,
            "chatbot_exists": 'chatbot' in globals(),
            "debug_info": f"chatbot type: {type(chatbot)}"
        }
        print(f"❌ Returning 400 error: {error_response}")
        return jsonify(error_response), 400
    
    try:
        data = request.get_json()
        test_query = data.get('query', 'Hello, can you help me?')
        
        # Check if it's our simple agent or complex agent
        if hasattr(chatbot.enhanced_agent, 'respond'):
            # Simple agent
            system_content = "You are a helpful childcare support assistant. Respond naturally to user questions about childcare applications and flags."
            response_content = chatbot.enhanced_agent.respond(test_query, system_content)
            
            return jsonify({
                "query": test_query,
                "response": response_content,
                "agent_type": "simple",
                "success": bool(response_content)
            })
        
        elif hasattr(chatbot.enhanced_agent, 'run'):
            # Complex agent
            if not hasattr(DialogTape, '__init__'):
                return jsonify({
                    "error": "DialogTape not available for complex agent",
                    "agent_type": "complex_failed"
                }), 400
            
            system_content = "You are a helpful childcare support assistant. Respond naturally to user questions."
            tape = DialogTape(context=None, steps=[SystemStep(content=system_content)])
            tape = tape.append(UserStep(content=test_query))
            
            response_content = ""
            event_count = 0
            
            for event in chatbot.enhanced_agent.run(tape):
                event_count += 1
                if event.final_tape:
                    final_tape = event.final_tape
                    last_step = final_tape.steps[-1]
                    if hasattr(last_step, 'content'):
                        response_content = last_step.content
                    break
            
            return jsonify({
                "query": test_query,
                "response": response_content,
                "events_processed": event_count,
                "agent_type": "complex",
                "success": bool(response_content)
            })
        
        else:
            return jsonify({
                "error": "Unknown agent type",
                "agent_type": str(type(chatbot.enhanced_agent)),
                "has_respond": hasattr(chatbot.enhanced_agent, 'respond'),
                "has_run": hasattr(chatbot.enhanced_agent, 'run')
            }), 400
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/chatbot/raw-test', methods=['POST'])
def test_raw_api():
    try:
        import requests
        
        print("🔍 Testing raw API call from endpoint...")
        
        api_key = "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"
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = request.get_json()
        test_query = data.get('query', 'Hello, can you help me with childcare applications?')
        
        payload = {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [
                {"role": "system", "content": "You are a helpful childcare support assistant for municipality workers. Explain application flags and regulations clearly."},
                {"role": "user", "content": test_query}
            ],
            "temperature": 0.1,
            "max_tokens": 300
        }
        
        print(f"🔍 Making raw API call with query: {test_query}")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"🔍 Raw API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"✅ Raw API success: {ai_response[:100]}...")
            
            return jsonify({
                "success": True,
                "query": test_query,
                "response": ai_response,
                "status_code": response.status_code,
                "method": "raw_api"
            })
        else:
            error_text = response.text
            print(f"❌ Raw API failed: {response.status_code} - {error_text}")
            
            return jsonify({
                "success": False,
                "error": f"API Error: {response.status_code}",
                "details": error_text,
                "status_code": response.status_code
            }), 400
            
    except Exception as e:
        import traceback
        error_details = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"❌ Raw API test exception: {error_details}")
        return jsonify(error_details), 500

@app.route('/api/review/<application_id>', methods=['POST'])
def review_application(application_id):
    try:
        data = request.get_json()
        decision = data.get('decision')
        reason = data.get('reason', '')
        notes = data.get('notes', '')
        
        if decision not in ['APPROVED', 'REJECTED']:
            return jsonify({"error": "Decision must be 'APPROVED' or 'REJECTED'"}), 400
        
        app_index = None
        for i, app in enumerate(processed_applications):
            if app.application_id == application_id:
                app_index = i
                break
        
        if app_index is None:
            return jsonify({"error": "Application not found"}), 404
        
        processed_applications[app_index].reviewer_decision = decision
        processed_applications[app_index].reviewer_reason = reason
        processed_applications[app_index].reviewer_notes = notes
        processed_applications[app_index].review_timestamp = datetime.now().isoformat()
        
        save_processing_results()
        
        return jsonify({
            "application_id": application_id,
            "decision": decision,
            "reason": reason,
            "notes": notes,
            "review_timestamp": processed_applications[app_index].review_timestamp
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/contact/<application_id>', methods=['POST'])
def contact_applicant(application_id):
    try:
        data = request.get_json()
        email = data.get('email', '')
        message = data.get('message', '')
        
        if not email or not message:
            return jsonify({"error": "Email and message are required"}), 400
        
        app_index = None
        for i, app in enumerate(processed_applications):
            if app.application_id == application_id:
                app_index = i
                break
        
        if app_index is None:
            return jsonify({"error": "Application not found"}), 404
        
        processed_applications[app_index].contact_email = email
        
        print(f"Contact attempt for {application_id}: {email} - {message}")
        
        save_processing_results()
        
        return jsonify({
            "application_id": application_id,
            "email": email,
            "message": "Contact logged successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze/<application_id>', methods=['POST'])
def analyze_application(application_id):
    try:

        app_index = None
        for i, app in enumerate(processed_applications):
            if app.application_id == application_id:
                app_index = i
                break
        
        if app_index is None:
            return jsonify({"error": "Application not found"}), 404
        

        app_data = asdict(processed_applications[app_index])
        llm_result = llm_provider.analyze_application(app_data)
        

        processed_applications[app_index].llm_analysis = llm_result['analysis']
        processed_applications[app_index].llm_decision = llm_result['decision']
        processed_applications[app_index].llm_reasoning = llm_result['reasoning']
        

        save_processing_results()
        
        return jsonify({
            "application_id": application_id,
            "llm_analysis": llm_result['analysis'],
            "llm_decision": llm_result['decision'],
            "llm_reasoning": llm_result['reasoning']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Childcare Subsidy Applications Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
            color: #1d1d1f;
            font-size: 32px;
            font-weight: 600;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stat-card {
            background: #007aff;
            color: white;
            padding: 20px;
            border-radius: 8px;
            flex: 1;
            min-width: 150px;
        }
        .stat-card:nth-child(2) {
            background: #ff8c00;
        }
        .stat-card:nth-child(3) {
            background: #28a745;
        }
        .stat-card:nth-child(4) {
            background: #ff3b30;
        }
        .stat-card h3 {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
        }
        .stat-card p {
            margin: 5px 0 0 0;
            opacity: 0.9;
        }
        .applications-grid {
            display: grid;
            gap: 16px;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
        }
        .application-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
        }
        .application-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        .app-id {
            font-size: 18px;
            font-weight: 600;
            color: #1d1d1f;
            margin-bottom: 8px;
        }
        .app-hash {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 12px;
            color: #8e8e93;
            margin-bottom: 12px;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-critical {
            background: #ff3b30;
            color: white;
        }
        .status-warning {
            background: #ff9500;
            color: white;
        }
        .status-review {
            background: #ff8c00;
            color: white;
        }
        .status-approved {
            background: #28a745;
            color: white;
        }
        .status-rejected {
            background: #ff3b30;
            color: white;
        }
        .status-summary {
            background: #007aff;
            color: white;
        }
        .flags {
            margin-top: 12px;
        }
        .flag {
            display: inline-block;
            background: #f2f2f7;
            color: #1d1d1f;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            margin: 2px 4px 2px 0;
        }
        .income-info {
            color: #8e8e93;
            font-size: 14px;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Childcare Subsidy Applications</h1>
        <div class="stats">
            <div class="stat-card">
                <h3>{{ applications|length }}</h3>
                <p>Total Applications</p>
            </div>
            <div class="stat-card">
                <h3>{{ pending_count }}</h3>
                <p>Pending Review</p>
            </div>
            <div class="stat-card">
                <h3>{{ approved_count }}</h3>
                <p>Approved</p>
            </div>
            <div class="stat-card">
                <h3>{{ rejected_count }}</h3>
                <p>Rejected</p>
            </div>
        </div>
    </div>

    <div class="applications-grid">
        {% for app in applications %}
        <div class="application-card" onclick="window.location.href='/application/{{ app.application_id }}'">
            <div class="app-id">{{ app.application_id }}</div>
            
            {% if app.eligibility_assessment == 'APPROVED' or app.reviewer_decision == 'APPROVED' %}
                <span class="status-badge status-approved">✅ Approved</span>
            {% elif app.reviewer_decision == 'REJECTED' %}
                <span class="status-badge status-rejected">❌ Rejected</span>
            {% elif app.llm_analysis %}
                <span class="status-badge status-summary">🤖 AI Summary Ready</span>
            {% else %}
                <span class="status-badge status-review">⏳ Pending Review</span>
            {% endif %}
            
            <div class="income-info">
                Income: ${{ "{:,}".format(app.household_income) }} | 
                Children: {{ app.num_children }} | 
                Hours: {{ app.childcare_hours_requested }}
            </div>
            
            <div class="flags">
                {% set flags_dict = app.validation_flags.__dict__ %}
                {% for key, value in flags_dict.items() %}
                    {% if value and key != 'missing_application_id' %}
                        <span class="flag">{{ key.replace('_', ' ').title() }}</span>
                    {% endif %}
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
'''

DETAIL_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Application {{ application.application_id }} - Detail</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            margin-bottom: 30px;
        }
        .back-button {
            background: #007aff;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 14px;
            display: inline-block;
            margin-bottom: 20px;
        }
        .back-button:hover {
            background: #0056cc;
        }
        .header h1 {
            margin: 0;
            color: #1d1d1f;
            font-size: 28px;
            font-weight: 600;
        }
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        .section {
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        }
        .section h2 {
            margin: 0 0 20px 0;
            color: #1d1d1f;
            font-size: 20px;
            font-weight: 600;
        }
        .field {
            margin-bottom: 16px;
        }
        .field-label {
            font-weight: 600;
            color: #1d1d1f;
            margin-bottom: 4px;
        }
        .field-value {
            color: #6e6e73;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 20px;
        }
        .status-critical {
            background: #ff3b30;
            color: white;
        }
        .status-warning {
            background: #ff9500;
            color: white;
        }
        .status-review {
            background: #007aff;
            color: white;
        }
        .flags-grid {
            display: grid;
            gap: 8px;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }
        .flag-item {
            display: flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
        }
        .flag-true {
            background: #ffebee;
            color: #d32f2f;
        }
        .flag-false {
            background: #e8f5e8;
            color: #2e7d32;
        }
        .flag-icon {
            margin-right: 8px;
            font-weight: bold;
        }
        .notes-list {
            list-style: none;
            padding: 0;
        }
        .notes-list li {
            background: #f2f2f7;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .hash-code {
            font-family: 'SF Mono', Monaco, monospace;
            background: #f2f2f7;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <a href="/" class="back-button">← Back to Dashboard</a>
    
    <div class="header">
        <h1>Application {{ application.application_id }}</h1>
        
        {% if application.eligibility_assessment == 'APPROVED' or application.reviewer_decision == 'APPROVED' %}
            <div class="status-badge status-approved">✅ Approved</div>
        {% elif application.reviewer_decision == 'REJECTED' %}
            <div class="status-badge status-rejected">❌ Rejected</div>
        {% else %}
            <div class="status-badge status-review">⏳ Pending Review</div>
        {% endif %}
    </div>

    <div class="content">
        <div class="section">
            <h2>Application Details</h2>
            <div class="field">
                <div class="field-label">Household Income</div>
                <div class="field-value">${{ "{:,}".format(application.household_income) }}</div>
            </div>
            <div class="field">
                <div class="field-label">Employment Status</div>
                <div class="field-value">{{ application.employment_status.title() }}</div>
            </div>
            <div class="field">
                <div class="field-label">Number of Children</div>
                <div class="field-value">{{ application.num_children }}</div>
            </div>
            <div class="field">
                <div class="field-label">Child Ages</div>
                <div class="field-value">{{ application.child_ages | join(', ') if application.child_ages else 'Not specified' }}</div>
            </div>
            <div class="field">
                <div class="field-label">Childcare Hours Requested</div>
                <div class="field-value">{{ application.childcare_hours_requested }} hours/month</div>
            </div>
            <div class="field">
                <div class="field-label">Housing Situation</div>
                <div class="field-value">{{ application.housing_situation.replace('_', ' ').title() }}</div>
            </div>
            <div class="field">
                <div class="field-label">Partner Employed</div>
                <div class="field-value">{{ 'Yes' if application.partner_employed else 'No' }}</div>
            </div>
        </div>

        <div class="section">
            <h2>Validation Flags</h2>
            <div class="flags-grid">
                {% for key, value in flags_dict.items() %}
                <div class="flag-item {{ 'flag-true' if value else 'flag-false' }}">
                    <span class="flag-icon">{{ '⚠️' if value else '✅' }}</span>
                    {{ key.replace('_', ' ').title() }}
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>AI-Generated Summary</h2>
            {% if application.llm_analysis %}
                <div class="llm-analysis-result">
                    <div class="field">
                        <div class="field-label">Professional Summary</div>
                        <div class="field-value summary-content">{{ application.llm_analysis | markdown_to_html | safe }}</div>
                    </div>
                    {% if application.llm_decision %}
                    <div class="field">
                        <div class="field-label">AI Decision</div>
                        <div class="field-value">{{ application.llm_decision }}</div>
                    </div>
                    {% endif %}
                    {% if application.llm_reasoning %}
                    <div class="field">
                        <div class="field-label">AI Reasoning</div>
                        <div class="field-value">{{ application.llm_reasoning }}</div>
                    </div>
                    {% endif %}
                    <button onclick="analyzeLLM('{{ application.application_id }}')" class="analyze-button" style="margin-top: 12px;">
                        🔄 Regenerate Summary
                    </button>
                </div>
            {% else %}
                <button id="analyze-btn" onclick="analyzeLLM('{{ application.application_id }}')" class="analyze-button">
                    🤖 Generate AI Summary
                </button>
                <div id="llm-results" style="display: none; margin-top: 16px;">
                </div>
            {% endif %}
        </div>

        <div class="section">
            <h2>Review Status</h2>
            {% if application.reviewer_decision %}
                <div class="review-completed">
                    <div class="field">
                        <div class="field-label">Final Decision</div>
                        <div class="field-value decision-{{ application.reviewer_decision.lower() }}">
                            {% if application.reviewer_decision == 'APPROVED' %}
                                ✅ APPROVED
                            {% else %}
                                ❌ REJECTED
                            {% endif %}
                        </div>
                    </div>
                    {% if application.reviewer_reason %}
                    <div class="field">
                        <div class="field-label">Decision Reason</div>
                        <div class="field-value">{{ application.reviewer_reason }}</div>
                    </div>
                    {% endif %}
                    {% if application.reviewer_notes %}
                    <div class="field">
                        <div class="field-label">Additional Notes</div>
                        <div class="field-value">{{ application.reviewer_notes }}</div>
                    </div>
                    {% endif %}
                    <div class="field">
                        <div class="field-label">Review Date</div>
                        <div class="field-value">{{ application.review_timestamp }}</div>
                    </div>
                </div>
            {% elif application.eligibility_assessment == 'REQUIRES_REVIEW' %}
                <div class="review-form">
                    <div class="field">
                        <label for="reviewer-reason" class="field-label">Review Reason:</label>
                        <textarea id="reviewer-reason" placeholder="Enter reason for decision..." class="review-textarea" style="min-height: 60px;"></textarea>
                    </div>
                    <div class="field">
                        <label for="reviewer-notes" class="field-label">Additional Notes:</label>
                        <textarea id="reviewer-notes" placeholder="Additional notes (optional)..." class="review-textarea" style="min-height: 60px;"></textarea>
                    </div>
                    <div class="review-buttons">
                        <button onclick="reviewApplication('APPROVED')" class="review-btn approve-btn">✅ Approve</button>
                        <button onclick="reviewApplication('REJECTED')" class="review-btn reject-btn">❌ Reject</button>
                    </div>
                </div>
            {% elif application.eligibility_assessment == 'APPROVED' %}
                <div class="auto-approved">
                    <p><strong>Assessment Date:</strong> {{ application.processed_timestamp[:19].replace('T', ' ') }}</p>
                </div>
                
                <div style="margin-top: 20px;">
                    <h3>Override Auto-Approval</h3>
                    <p style="color: #666; margin-bottom: 15px;">If you need to change the status of this auto-approved application, provide a reason below.</p>
                    <div class="review-form">
                        <div class="field">
                            <label for="reviewer-reason" class="field-label">Reason for Status Change:</label>
                            <textarea id="reviewer-reason" placeholder="Please provide a reason for changing the status..." class="review-textarea" style="min-height: 60px;"></textarea>
                        </div>
                        <div class="field">
                            <label for="reviewer-notes" class="field-label">Additional Notes:</label>
                            <textarea id="reviewer-notes" placeholder="Additional notes (optional)..." class="review-textarea" style="min-height: 60px;"></textarea>
                        </div>
                        <div class="review-buttons" >
                            <button onclick="reviewApplication('APPROVED')" class="review-btn approve-btn">✅ Keep Approved</button>
                            <button onclick="reviewApplication('REJECTED')" class="review-btn reject-btn">❌ Reject Application</button>
                        </div>
                    </div>
                </div>
            {% else %}
                <div class="pending-review">
                    <p>This application has not been reviewed yet.</p>
                </div>
            {% endif %}
        </div>
        
        {% if not application.reviewer_decision %}
        <div class="section">
            <h2>Contact Applicant</h2>
            <div class="contact-form">
                <div class="field">
                    <label for="contact-email" class="field-label">Email:</label>
                    <input type="email" id="contact-email" placeholder="applicant@example.com" class="contact-input">
                </div>
                <div class="field">
                    <label for="contact-message" class="field-label">Message:</label>
                    <textarea id="contact-message" placeholder="Enter your message..." class="contact-textarea" style="min-height: 80px;"></textarea>
                </div>
                <button onclick="contactApplicant()" class="contact-btn">📧 Send Message</button>
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>Chatbot Assistant</h2>
            <div class="chatbot-container">
                <div class="chatbot-controls">
                    <button onclick="testChatbot()" class="test-btn">🧪 Test LLM Connection</button>
                    <button onclick="testDirectLLM()" class="test-btn">⚡ Test Direct LLM</button>
                    <button onclick="testRawAPI()" class="test-btn">🌐 Test Raw API</button>
                </div>
                <div id="chat-messages" class="chat-messages"></div>
                <div class="chat-input-container">
                    <input type="text" id="chat-input" placeholder="Ask about this application..." class="chat-input">
                    <button onclick="sendChatMessage()" class="chat-send-btn">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function analyzeLLM(applicationId) {
            const btn = document.getElementById('analyze-btn');
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            try {
                const response = await fetch(`/api/analyze/${applicationId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('llm-results').innerHTML = `
                        <div class="field">
                            <div class="field-label">Analysis</div>
                            <div class="field-value">${result.llm_analysis}</div>
                        </div>
                        <div class="field">
                            <div class="field-label">Decision</div>
                            <div class="field-value">${result.llm_decision}</div>
                        </div>
                        <div class="field">
                            <div class="field-label">Reasoning</div>
                            <div class="field-value">${result.llm_reasoning}</div>
                        </div>
                    `;
                    document.getElementById('llm-results').style.display = 'block';
                    btn.style.display = 'none';
                } else {
                    alert('Error: ' + result.error);
                    btn.disabled = false;
                    btn.textContent = 'Analyze with LLM';
                }
            } catch (error) {
                alert('Error analyzing application: ' + error.message);
                btn.disabled = false;
                btn.textContent = 'Analyze with LLM';
            }
        }

        async function reviewApplication(decision) {
            const reason = document.getElementById('reviewer-reason').value.trim();
            const notes = document.getElementById('reviewer-notes').value.trim();
            
            if (!reason) {
                alert('Please provide a reason for your decision.');
                return;
            }
            
            try {
                const response = await fetch(`/api/review/{{ application.application_id }}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        decision: decision,
                        reason: reason,
                        notes: notes
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    alert(`Application ${decision.toLowerCase()} successfully!`);
                    location.reload();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error reviewing application: ' + error.message);
            }
        }
        
        async function contactApplicant() {
            const email = document.getElementById('contact-email').value.trim();
            const message = document.getElementById('contact-message').value.trim();
            
            if (!email || !message) {
                alert('Please provide both email and message.');
                return;
            }
            
            try {
                const response = await fetch(`/api/contact/{{ application.application_id }}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        email: email,
                        message: message
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    alert('Message sent successfully!');
                    document.getElementById('contact-email').value = '';
                    document.getElementById('contact-message').value = '';
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error sending message: ' + error.message);
            }
        }

        async function testChatbot() {
            const messagesContainer = document.getElementById('chat-messages');
            
            messagesContainer.innerHTML += `
                <div class="chat-message system-message">
                    <strong>System:</strong> Testing LLM connection...
                </div>
            `;
            
            try {
                const response = await fetch('/api/chatbot/test', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: 'Hello, can you help me understand application flags?'
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    messagesContainer.innerHTML += `
                        <div class="chat-message system-message">
                            <strong>Test Results:</strong><br>
                            TapeAgents Available: ${result.status.tapeagents_available}<br>
                            Enhanced Agent: ${result.status.enhanced_agent_available}<br>
                            Agent Type: ${result.agent_type}<br>
                            <br><strong>Test Response:</strong><br>
                            <pre>${result.response}</pre>
                        </div>
                    `;
                } else {
                    messagesContainer.innerHTML += `
                        <div class="chat-message error-message">
                            <strong>Test Failed:</strong> ${result.error}<br>
                            ${result.traceback ? '<details><summary>Details</summary><pre>' + result.traceback + '</pre></details>' : ''}
                        </div>
                    `;
                }
                
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                
            } catch (error) {
                messagesContainer.innerHTML += `
                    <div class="chat-message error-message">
                        <strong>Test Error:</strong> ${error.message}
                    </div>
                `;
            }
        }

        async function testDirectLLM() {
            const messagesContainer = document.getElementById('chat-messages');
            
            messagesContainer.innerHTML += `
                <div class="chat-message system-message">
                    <strong>Direct LLM Test:</strong> Testing LLM with fresh conversation...
                </div>
            `;
            
            try {
                const response = await fetch('/api/chatbot/direct', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: 'What are childcare application flags and what do they mean?'
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    messagesContainer.innerHTML += `
                        <div class="chat-message system-message">
                            <strong>Direct LLM Results:</strong><br>
                            Success: ${result.success}<br>
                            Events Processed: ${result.events_processed}<br>
                            <br><strong>Direct LLM Response:</strong><br>
                            <pre>${result.response}</pre>
                        </div>
                    `;
                } else {
                    messagesContainer.innerHTML += `
                        <div class="chat-message error-message">
                            <strong>Direct Test Failed:</strong> ${result.error}<br>
                            ${result.traceback ? '<details><summary>Details</summary><pre>' + result.traceback + '</pre></details>' : ''}
                        </div>
                    `;
                }
                
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                
            } catch (error) {
                messagesContainer.innerHTML += `
                    <div class="chat-message error-message">
                        <strong>Direct Test Error:</strong> ${error.message}
                    </div>
                `;
            }
        }

        async function testRawAPI() {
            const messagesContainer = document.getElementById('chat-messages');
            
            messagesContainer.innerHTML += `
                <div class="chat-message system-message">
                    <strong>Raw API Test:</strong> Testing direct OpenRouter API call...
                </div>
            `;
            
            try {
                const response = await fetch('/api/chatbot/raw-test', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: 'What are childcare application flags and what do they mean?'
                    })
                });
                
                const result = await response.json();
                
                if (response.ok && result.success) {
                    messagesContainer.innerHTML += `
                        <div class="chat-message system-message">
                            <strong>Raw API Success!</strong><br>
                            Status: ${result.status_code}<br>
                            Method: ${result.method}<br>
                            <br><strong>AI Response:</strong><br>
                            <pre>${result.response}</pre>
                        </div>
                    `;
                } else {
                    messagesContainer.innerHTML += `
                        <div class="chat-message error-message">
                            <strong>Raw API Failed:</strong><br>
                            Error: ${result.error}<br>
                            Status: ${result.status_code || 'Unknown'}<br>
                            ${result.details ? '<br>Details: ' + result.details : ''}
                            ${result.traceback ? '<details><summary>Traceback</summary><pre>' + result.traceback + '</pre></details>' : ''}
                        </div>
                    `;
                }
                
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                
            } catch (error) {
                messagesContainer.innerHTML += `
                    <div class="chat-message error-message">
                        <strong>Raw API Test Error:</strong> ${error.message}
                    </div>
                `;
            }
        }

        async function sendChatMessage() {
            const input = document.getElementById('chat-input');
            const query = input.value.trim();
            
            if (!query) return;
            
            const messagesContainer = document.getElementById('chat-messages');
            
            // Add user message
            messagesContainer.innerHTML += `
                <div class="chat-message user-message">
                    <strong>You:</strong> ${query}
                </div>
            `;
            
            input.value = '';
            
            try {
                const response = await fetch('/api/chatbot', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: query,
                        application_id: '{{ application.application_id }}'
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    messagesContainer.innerHTML += `
                        <div class="chat-message bot-message">
                            <strong>Assistant:</strong> <pre>${result.response}</pre>
                        </div>
                    `;
                } else {
                    messagesContainer.innerHTML += `
                        <div class="chat-message error-message">
                            <strong>Error:</strong> ${result.error}
                        </div>
                    `;
                }
                
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                
            } catch (error) {
                messagesContainer.innerHTML += `
                    <div class="chat-message error-message">
                        <strong>Error:</strong> ${error.message}
                    </div>
                `;
            }
        }

        // Allow Enter key to send messages
        document.getElementById('chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    </script>

    <style>
        .analyze-button {
            background: #007aff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        .analyze-button:hover {
            background: #0056cc;
        }
        .analyze-button:disabled {
            background: #8e8e93;
            cursor: not-allowed;
        }
        .chatbot-container {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            background: #fafafa;
        }
        .chat-messages {
            max-height: 300px;
            overflow-y: auto;
            margin-bottom: 12px;
            padding: 8px;
            background: white;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }
        .chat-message {
            margin-bottom: 8px;
            padding: 8px;
            border-radius: 4px;
        }
        .user-message {
            background: #e3f2fd;
            text-align: right;
        }
        .bot-message {
            background: #f5f5f5;
        }
        .bot-message pre {
            white-space: pre-wrap;
            font-family: inherit;
            margin: 4px 0 0 0;
        }
        .error-message {
            background: #ffebee;
            color: #d32f2f;
        }
        .chat-input-container {
            display: flex;
            gap: 8px;
        }
        .chat-input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            font-size: 14px;
        }
        .chat-send-btn {
            background: #007aff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .chat-send-btn:hover {
            background: #0056cc;
        }
        .review-form {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .review-textarea {
            width: 95%;
            min-height: 80px;
            padding: 12px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
        }
        .review-buttons {
            display: flex;
            gap: 12px;
            margin-top: 16px;
        }
        .review-btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        .approve-btn {
            background: #34c759;
            color: white;
        }
        .approve-btn:hover {
            background: #28a745;
        }
        .reject-btn {
            background: #ff3b30;
            color: white;
        }
        .reject-btn:hover {
            background: #dc3545;
        }
        .review-result {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .decision-approved {
            color: #34c759;
            font-weight: 600;
        }
        .decision-rejected {
            color: #ff3b30;
            font-weight: 600;
        }
        .contact-form {
            background: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #b3d9ff;
        }
        .contact-input {
            width: 95%;
            padding: 8px 12px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            font-size: 14px;
        }
        .contact-textarea {
            width: 95%;
            min-height: 100px;
            padding: 12px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
        }
        .contact-btn {
            background: #007aff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            margin-top: 12px;
        }
        .contact-btn:hover {
            background: #0056cc;
        }
        .review-completed {
            background: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #007aff;
        }
        .pending-review {
            background: #fff8e1;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ffa726;
            text-align: center;
            color: #e65100;
        }
        .status-approved {
            background: #28a745;
            color: white;
        }
        .status-rejected {
            background: #ff3b30;
            color: white;
        }
        .status-review {
            background: #ff8c00;
            color: white;
        }
        .llm-analysis-result {
            background: #f8f9ff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #d4e6ff;
        }
        .summary-content {
            line-height: 1.6;
            font-size: 14px;
            background: white;
            padding: 16px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            max-height: 400px;
            overflow-y: auto;
        }
        .summary-content h3 {
            margin: 16px 0 8px 0;
            color: #1d1d1f;
            font-size: 16px;
            font-weight: 600;
        }
        .summary-content ul {
            margin: 8px 0;
            padding-left: 20px;
        }
        .summary-content li {
            margin: 4px 0;
        }
        .summary-content strong {
            color: #1d1d1f;
            font-weight: 600;
        }
        .chatbot-controls {
            margin-bottom: 12px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .test-btn {
            background: #17a2b8;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .test-btn:hover {
            background: #138496;
        }
        .system-message {
            background: #e7f3ff;
            border-left: 4px solid #007bff;
        }
    </style>
</body>
</html>
'''

def main():
    load_and_process_applications()
    print("Starting web application on http://localhost:8082")
    app.run(debug=True, host='0.0.0.0', port=8082)

if __name__ == "__main__":
    main()
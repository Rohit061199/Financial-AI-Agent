from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

#Web search agent
webserach_agent=Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include source"],
    show_tool_calls=True,
    markdown=True
)

#Financial Agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use Tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

#multi_ai_agent=Agent(
#    team=[webserach_agent,finance_agent],
#    instructions=["Always include source","Use Tables to display the data"],
#    model = Groq(id="llama-3.1-70b-versatile"),
#    show_tool_calls=True,
#    markdown=True
#)

app=Playground(agents=[finance_agent,webserach_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)

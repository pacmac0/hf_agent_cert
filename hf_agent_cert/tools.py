import sympy
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from google.genai import types


def search_wikipedia(query: str, top_k_results: int = 3, doc_content_chars_max: int = 2000) -> str:
    """Search Wikipedia for information about a specific topic.
    
    Performs a structured search of Wikipedia's knowledge base and returns
    formatted results with clear delimiters for easy parsing.
    
    Args:
        query: The search query for Wikipedia (e.g., "Albert Einstein", "quantum mechanics").
        top_k_results: Maximum number of articles to retrieve. Defaults to 3.
        doc_content_chars_max: Maximum characters per article. Defaults to 2000.
        
    Returns:
        Formatted string containing search results with headers and delimiters,
        or error message if search fails.
        
    Example:
        >>> result = search_wikipedia("photosynthesis")
        >>> print(result)
        === WIKIPEDIA SEARCH: photosynthesis ===
        Photosynthesis is the process by which plants...
        === END WIKIPEDIA RESULTS ===
    """
    try:
        wrapper = WikipediaAPIWrapper(top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max)
        result = wrapper.run(query)
        
        # Format the result with clear structure
        formatted_result = f"""=== WIKIPEDIA SEARCH: {query} ===

{result}

=== END WIKIPEDIA RESULTS ==="""
        
        return formatted_result
    except Exception as e:
        return f"❌ ERROR - Wikipedia search failed: {str(e)}"

  
def search_arxiv(query: str, max_results: int = 3) -> str:
    """Search ArXiv repository for academic research papers.
    
    Queries the ArXiv.org database for scholarly articles matching the search terms
    and returns formatted results with metadata including titles, authors, dates, and abstracts.
    
    Args:
        query: Search query for ArXiv papers (e.g., "transformer architecture", "neural networks").
        max_results: Maximum number of papers to return. Defaults to 3.
        
    Returns:
        Formatted string containing search results with paper metadata and clear delimiters,
        or error message if no papers found or search fails.
        
    Example:
        >>> result = search_arxiv("attention mechanism", max_results=2)
        >>> print(result)
        === ARXIV SEARCH: attention mechanism ===
        Found 2 paper(s)
        
        📄 PAPER 1:
        Title: Attention Is All You Need
        Authors: Ashish Vaswani, et al.
        Published: 2017-06-12
        Abstract: The dominant sequence transduction models...
    """
    try:
        loader = ArxivLoader(query=query, max_results=max_results)
        docs = loader.load()
        
        if not docs:
            return f"❌ No ArXiv papers found for query: '{query}'"
        
        # Structured formatting for better parsing
        results = [f"=== ARXIV SEARCH: {query} ==="]
        results.append(f"Found {len(docs)} paper(s)")
        results.append("")
        
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            title = metadata.get("Title", "Unknown Title")
            authors = metadata.get("Authors", "Unknown Authors")
            published = metadata.get("Published", "Unknown Date")
            
            results.append(f"""📄 PAPER {i}:
Title: {title}
Authors: {authors}
Published: {published}
Abstract: {doc.page_content}
---""")
        
        results.append("=== END ARXIV RESULTS ===")
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ ERROR - ArXiv search failed: {str(e)}"


def calculate(expression: str) -> str:
    """Perform mathematical calculations using arithmetic and symbolic computation.
    
    Intelligently handles simple arithmetic operations and complex symbolic mathematics
    using SymPy. Automatically detects expression complexity and chooses the
    appropriate evaluation method.
    
    Args:
        expression: Mathematical expression to evaluate. Examples:
            - Arithmetic: "2 + 3 * 4", "100 / 5", "(10 + 2) * 3"
            - Symbolic: "sin(pi/2)", "sqrt(16)", "log(e)", "x**2 + 2*x + 1"
        
    Returns:
        Formatted calculation result with original expression, computed result,
        and calculation type, or error message if calculation fails.
        
    Example:
        >>> result = calculate("sin(pi/4) + cos(pi/4)")
        >>> print(result)
        🧮 CALCULATION RESULT:
        Expression: sin(pi/4) + cos(pi/4)
        Simplified: sqrt(2)
        Type: Symbolic mathematics
        
    Note:
        Only mathematical operations are supported for security.
    """
    try:
        # Simple arithmetic first
        if all(c in "0123456789+-*/.() " for c in expression):
            result = eval(expression)
            return f"""🧮 CALCULATION RESULT:
Expression: {expression}
Result: {result}
Type: Arithmetic calculation"""
        
        # Symbolic math with SymPy for more complex expressions
        result = sympy.sympify(expression)
        evaluated = sympy.simplify(result)
        
        return f"""🧮 CALCULATION RESULT:
Expression: {expression}
Simplified: {evaluated}
Type: Symbolic mathematics"""
        
    except Exception as e:
        return f"❌ CALCULATION ERROR: {str(e)}\nExpression: {expression}"


def solve_equation(equation: str, variable: str = "x") -> str:
    """Solve mathematical equations symbolically for specified variables.
    
    Uses SymPy's symbolic mathematics engine to solve linear, quadratic, polynomial,
    trigonometric, exponential, and logarithmic equations. Automatically handles
    multiple solutions and formats results clearly.
    
    Args:
        equation: Mathematical equation to solve, expressed as left side of "equation = 0".
            Examples: "2*x + 5", "x**2 - 4", "sin(x) - 1/2"
        variable: The variable to solve for. Defaults to "x".
        
    Returns:
        Formatted equation solution with original equation, variable, and solutions,
        or error message if solving fails.
        
    Example:
        >>> result = solve_equation("x**2 - 9", "x")
        >>> print(result)
        🔧 EQUATION SOLUTION:
        Equation: x**2 - 9 = 0
        Variable: x
        Solution: x = [-3, 3]
        
    Note:
        Equation assumed to equal zero. Complex solutions included in results.
    """
    try:
        x = sympy.Symbol(variable)
        eq = sympy.sympify(equation)
        solution = sympy.solve(eq, x)
        
        # Format solution nicely
        if isinstance(solution, list):
            if len(solution) == 0:
                solution_text = "No solutions found"
            elif len(solution) == 1:
                solution_text = f"{variable} = {solution[0]}"
            else:
                solution_text = f"{variable} = {solution}"
        else:
            solution_text = f"{variable} = {solution}"
        
        return f"""🔧 EQUATION SOLUTION:
Equation: {equation} = 0
Variable: {variable}
Solution: {solution_text}"""
        
    except Exception as e:
        return f"❌ EQUATION ERROR: {str(e)}\nEquation: {equation}\nVariable: {variable}"


# List of minimal external tools (Gemini handles web search, URLs, images, audio natively)
# Note: Google Search and Code Execution are enabled via model configuration, not as tools
CUSTOM_TOOLS = [
    search_wikipedia,    # Structured encyclopedia data
    search_arxiv,        # Academic paper search
    calculate,           # Advanced symbolic math
    solve_equation       # Equation solving
]

GOOGLE_GEMINI_TOOLS = [
    {"url_context": {}},
    types.Tool(google_search=types.GoogleSearch())
]



# Test calls for all tools
def test_tools() -> None:
    """Test all custom tools with sample data to verify functionality and integration.
    
    This function performs comprehensive testing of all available custom tools
    by executing them with representative test cases. It provides immediate
    feedback on tool availability, proper configuration, and basic functionality.
    Useful for debugging, deployment verification, and development setup validation.
    
    The function tests:
    - Wikipedia search with a well-known topic
    - ArXiv search with a common research term  
    - Mathematical calculator with arithmetic expression
    - Equation solver with quadratic equation
    
    Returns:
        None: Results are printed directly to stdout with success/failure indicators.
        
    Example:
        >>> test_tools()
        ============================================================
        TESTING ALL TOOLS
        ============================================================
        
        1. Testing Wikipedia search...
        ✅ Wikipedia search successful
        Result preview: === WIKIPEDIA SEARCH: Albert Einstein ===...
        
    Note:
        Requires active internet connection for Wikipedia and ArXiv searches.
        SymPy must be properly installed for mathematical operations.
    """
    print("=" * 60)
    print("TESTING ALL TOOLS")
    print("=" * 60)
    
    # Test Wikipedia search
    print("\n1. Testing Wikipedia search...")
    try:
        result = search_wikipedia("Albert Einstein")
        print(f"✅ Wikipedia search successful")
        print(f"Result preview: {result[:200]}...")
    except Exception as e:
        print(f"❌ Wikipedia search failed: {e}")
    
    # Test ArXiv search
    print("\n2. Testing ArXiv search...")
    try:
        result = search_arxiv("machine learning")
        print(f"✅ ArXiv search successful")
        print(f"Result preview: {result[:200]}...")
    except Exception as e:
        print(f"❌ ArXiv search failed: {e}")
    
    # Test calculator
    print("\n3. Testing calculator...")
    try:
        result = calculate("2 + 2 * 3")
        print(f"✅ Calculator successful")
        print(f"Result: {result}")
    except Exception as e:
        print(f"❌ Calculator failed: {e}")
    
    # Test equation solver
    print("\n4. Testing equation solver...")
    try:
        result = solve_equation("x**2 - 4", "x")
        print(f"✅ Equation solver successful")
        print(f"Result: {result}")
    except Exception as e:
        print(f"❌ Equation solver failed: {e}")
    
    print("\n" + "=" * 60)
    print("TOOL TESTING COMPLETE")
    print("=" * 60)


# Uncomment the line below to run tests when this file is imported
if __name__ == "__main__":
    test_tools()


    
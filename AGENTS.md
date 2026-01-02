# AGENTS.md

## Build/Test Commands
```bash
# Install dependencies
uv sync

# Activate virtual env. 
. .venv/bin/activate

# Run main multi-agent system
python multi-agent-system.py

# Run test suite
python multi-agent-test.py

# Quick test of vector search tools
python -c "from multi_agent_test import test_tools; test_tools()"
```

## Code Style Guidelines

### Imports
- Use `uv` for dependency management
- Group imports: stdlib → third-party → local
- Use descriptive module comments (e.g., `# LangChain document loaders...`)

### Naming Conventions
- Functions: snake_case (e.g., `make_diabets_vector`)
- Variables: snake_case (e.g., `diabetes_vectorstore`)
- Constants: UPPER_CASE (e.g., `DIABETES_PDF_PATHS`)
- Classes: PascalCase (e.g., `RAGQA`, `SimpleQA`)

### DSPy Patterns
- Always configure `dspy.settings.configure(lm=lm)` after creating LM
- Use proper Signature classes, not string literals for ReAct
- Handle vector store persistence with FAISS `save_local`/`load_local`
- Add persistence checks before rebuilding vector databases

### Error Handling
- Wrap file operations in try/except blocks
- Check directory existence before loading vector stores
- Use `allow_dangerous_deserialization=True` for FAISS loading
- Gracefully handle missing API keys and model availability

### MLflow Integration
- Note: mlflow.dspy.autolog() may not be available in all versions
- Set experiment name and tracking URI before autolog calls
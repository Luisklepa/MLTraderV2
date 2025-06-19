# Contributing to ML Trading System

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Code Style

We use the following code style conventions:

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable names
- Comment complex algorithms and business logic

## Project Structure

```
├── analysis/              # Analysis tools and scripts
├── backtest/             # Backtesting engine
├── config/               # Configuration files
├── data/                 # Data storage
├── indicators/           # Technical indicators
├── models/              # Model storage
├── scripts/             # Main execution scripts
├── strategies/          # Trading strategies
├── tests/              # Unit tests
└── utils/              # Utility functions
```

## Setting Up Development Environment

1. Create a virtual environment:
```bash
python -m venv venv311
source venv311/bin/activate  # Linux/Mac
venv311\\Scripts\\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ml_pipeline.py

# Run with coverage report
pytest --cov=./ tests/
```

## Documentation

- Keep README.md updated with any user-facing changes
- Update ARCHITECTURE.md for significant structural changes
- Document all new features and API changes
- Add docstrings to all new functions and classes

## Performance Considerations

When contributing optimizations:

1. Data Processing
- Use parallel processing where appropriate
- Implement caching for expensive operations
- Optimize memory usage for large datasets

2. Model Training
- Profile performance before optimization
- Consider both CPU and memory usage
- Document performance improvements

3. Risk Management
- Ensure real-time calculations are efficient
- Optimize portfolio calculations
- Profile risk metric computations

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the ARCHITECTURE.md if you're changing system design
3. Update the requirements.txt if you're adding dependencies
4. The PR will be merged once you have the sign-off of maintainers

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker]

We use GitHub issues to track public bugs. Report a bug by [opening a new issue]().

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under its MIT License. 
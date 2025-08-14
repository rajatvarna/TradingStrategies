# Quant Trading Platform

This project is a comprehensive platform for developing, backtesting, and deploying quantitative trading strategies. It features a powerful Python backtesting library, a full-featured web API for managing strategies, and an AI-powered chatbot for assistance.

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Services](#running-the-services)
  - [Using the API](#using-the-api)
- [Future Work](#future-work)

## Architecture

The platform is built with a modular architecture, consisting of three main components:

1.  **`quant_strategies` Library**: A core Python library for creating and backtesting quantitative trading strategies. It supports both signal-based and event-driven (order-based) backtesting.

2.  **`web_app` (Flask API)**: A Flask-based web application that provides a REST API for:
    *   User management (registration, login)
    *   CRUD operations for trading strategies
    *   Running backtests, optimizations, and walk-forward analyses
    *   Paper trading
    *   Portfolio management

3.  **`chatbot_service` (FastAPI)**: An AI assistant powered by FastAPI and LangChain. It is designed to answer questions about trading strategies and the platform. (Currently under development).

## Features

*   **Flexible Backtesting**: Backtest strategies using either simple signals or complex order logic.
*   **Advanced Analytics**: Perform parameter optimization and walk-forward analysis to validate your strategies.
*   **RESTful API**: A full-featured API to manage users, strategies, and backtests programmatically.
*   **User and Strategy Management**: A database-backed system for storing user accounts and their trading strategies.
*   **Paper Trading**: Deploy your strategies in a simulated environment to track their performance in real-time.
*   **AI Assistant**: A chatbot to help you with your quantitative analysis.

## Getting Started

### Prerequisites

*   Python 3.10+
*   `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies for each service:**

    The project is divided into three components, each with its own dependencies.

    *   **Install the core `quant_strategies` library:**
        This project uses a `pyproject.toml` to define the core library. Install it in editable mode:
        ```bash
        pip install -e .
        ```

    *   **Web App:**
        ```bash
        pip install -r web_app/requirements.txt
        ```

    *   **Chatbot Service:**
        ```bash
        pip install -r chatbot_service/requirements.txt
        ```

3.  **Initialize the Database:**

    The web app uses a SQLite database. To create it and run the initial migrations, run the following commands from the root directory:

    ```bash
    export FLASK_APP="web_app.app:create_app()"
    flask db upgrade
    ```
    *(Note: The migrations directory is already included. If you need to create migrations from scratch, you would run `flask db init` and `flask db migrate` before the upgrade.)*

## Usage

### Running the Services

*   **Flask Web API:**
    ```bash
    FLASK_APP="web_app.app:create_app()" flask run
    ```
    The API will be available at `http://127.0.0.1:5000`.

*   **Chatbot Service:**
    ```bash
    uvicorn chatbot_service.main:app --reload
    ```
    The chatbot service will be available at `http://127.0.0.1:8000`. You can see the auto-generated documentation at `http://127.0.0.1:8000/docs`.

### Using the API

You can interact with the API using a tool like `curl` or Postman. Here is a simple workflow:

1.  **Register a new user:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"username": "testuser", "email": "test@example.com", "password": "password123"}' http://127.0.0.1:5000/api/register
    ```

2.  **Log in to get a JWT token:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"username": "testuser", "password": "password123"}' http://127.0.0.1:5000/api/login
    ```
    *(This will return a token that you need to include in the `Authorization` header for protected endpoints.)*

3.  **Create a strategy:**
    ```bash
    TOKEN="your_jwt_token_here"
    curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -d '{"name": "My First Strategy", "config": {"tickers": ["AAPL", "GOOG"], "start_date": "2020-01-01"}}' http://127.0.0.1:5000/api/strategies
    ```

## Future Work

This project is actively being developed. Some of the planned improvements include:

*   Adding comprehensive unit and integration tests.
*   Implementing the full RAG pipeline for the chatbot.
*   Developing a frontend application for a more user-friendly experience.
*   Refactoring the web app to improve modularity and remove circular dependencies.
*   Improving configuration management to avoid hardcoded secrets.

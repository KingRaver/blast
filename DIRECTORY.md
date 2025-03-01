blast/
│
├── README.md                   # Project overview and setup instructions
├── __init__.py                 # Python package initialization
├── architecture.txt            # This file - project structure documentation
├── get-pip.py                  # Python package management script
├── requirements.txt            # Project dependencies
│
├── logs/                       # Logging directory
│   ├── analysis/               # Specific analysis logs
│   │   └── market_analysis.log # Market-related log files
│   ├── claude.log              # Claude interaction logs
│   ├── coingecko.log           # CoinGecko API logs
│   └── ... (other log files)   # Additional log files
│
└── src/                        # Source code directory
    ├── __init__.py             # Package initialization
    ├── bot.py                  # Main bot implementation
    ├── config.py               # Configuration management
    │
    └── utils/                  # Utility modules
        ├── __init__.py         # Utility package initialization
        ├── browser.py          # Browser-related utilities
        ├── logger.py           # Logging utilities
        └── sheets_handler.py   # Spreadsheet handling utilities

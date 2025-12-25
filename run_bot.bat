@echo off
chcp 65001 > nul
echo ========================================
echo   RAG for Law - Telegram Bot Launcher
echo ========================================
echo.

REM Проверяем наличие виртуального окружения
if not exist ".venv\Scripts\activate.bat" (
    echo [ОШИБКА] Виртуальное окружение не найдено!
    echo Создайте его командой: python -m venv .venv
    echo.
    pause
    exit /b 1
)

REM Активируем виртуальное окружение
echo [1/3] Активация виртуального окружения...
call .venv\Scripts\activate.bat

REM Проверяем наличие .env файла
if not exist ".env" (
    echo [ОШИБКА] Файл .env не найден!
    echo Скопируйте .env.example в .env и укажите ваш TELEGRAM_BOT_TOKEN
    echo.
    pause
    exit /b 1
)

REM Проверяем наличие токена в .env
findstr /C:"TELEGRAM_BOT_TOKEN=your_bot_token_here" .env > nul
if %errorlevel% == 0 (
    echo [ОШИБКА] Токен бота не настроен!
    echo Откройте файл .env и замените "your_bot_token_here" на реальный токен от @BotFather
    echo.
    pause
    exit /b 1
)

echo [2/3] Проверка зависимостей...
python -c "import telegram; import dotenv" 2>nul
if %errorlevel% neq 0 (
    echo Устанавливаем недостающие зависимости...
    pip install -q python-telegram-bot python-dotenv
)

echo [3/3] Запуск бота...
echo.
echo ========================================
echo   Бот запущен! Нажмите Ctrl+C для остановки
echo ========================================
echo.

python bot.py

pause

#!/bin/bash

# LoRA Tuymans Monitoring Script
# Poskytuje detailní diagnostiku a monitoring aplikace

set -euo pipefail

# Barvy pro výstup
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging funkce
log_info() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"; }
log_error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" >&2; }

# Funkce pro kontrolu služeb
check_streamlit() {
    log_info "Kontrola Streamlit procesu..."
    
    if pgrep -f "streamlit.*8505" >/dev/null; then
        local pid=$(pgrep -f "streamlit.*8505")
        local memory=$(ps -p "$pid" -o rss= 2>/dev/null | awk '{print $1/1024 " MB"}' || echo "N/A")
        local cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | awk '{print $1"%"}' || echo "N/A")
        log_success "Streamlit běží (PID: $pid, Memory: $memory, CPU: $cpu)"
        return 0
    else
        log_error "Streamlit proces neběží!"
        return 1
    fi
}

check_port() {
    log_info "Kontrola portu 8505..."
    
    if command -v netstat >/dev/null; then
        if netstat -tuln 2>/dev/null | grep -q ":8505.*LISTEN"; then
            log_success "Port 8505 je aktivní a naslouchá"
            return 0
        fi
    fi
    
    if command -v ss >/dev/null; then
        if ss -tuln 2>/dev/null | grep -q ":8505.*LISTEN"; then
            log_success "Port 8505 je aktivní a naslouchá"
            return 0
        fi
    fi
    
    log_error "Port 8505 není dostupný!"
    return 1
}

check_health() {
    log_info "Kontrola health endpointu..."
    
    if command -v curl >/dev/null; then
        if curl -s -f http://localhost:8505/_stcore/health >/dev/null 2>&1; then
            log_success "Health endpoint odpovídá"
            return 0
        elif curl -s -f http://localhost:8505/ >/dev/null 2>&1; then
            log_warning "Hlavní stránka odpovídá, ale health endpoint ne"
            return 1
        fi
    fi
    
    log_error "Health endpoint neodpovídá!"
    return 1
}

check_gpu() {
    log_info "Kontrola GPU..."
    
    if command -v nvidia-smi >/dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
        if [[ "$gpu_info" != "N/A" ]]; then
            log_success "GPU je dostupná:"
            echo "$gpu_info" | while IFS=, read -r name memory_used memory_total utilization; do
                echo "  - $name: ${memory_used}MB/${memory_total}MB (${utilization}% využití)"
            done
            return 0
        fi
    fi
    
    log_warning "GPU není dostupná nebo nvidia-smi není nainstalováno"
    return 1
}

check_disk_space() {
    log_info "Kontrola diskového prostoru..."
    
    local disk_usage=$(df -h /app 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//' || echo "N/A")
    if [[ "$disk_usage" != "N/A" ]]; then
        if [[ $disk_usage -lt 80 ]]; then
            log_success "Diskový prostor: ${disk_usage}% využito"
        elif [[ $disk_usage -lt 90 ]]; then
            log_warning "Diskový prostor: ${disk_usage}% využito (blíží se limit)"
        else
            log_error "Diskový prostor: ${disk_usage}% využito (kritický stav!)"
        fi
    else
        log_warning "Nelze zjistit využití diskového prostoru"
    fi
}

check_memory() {
    log_info "Kontrola paměti..."
    
    if command -v free >/dev/null; then
        local memory_info=$(free -h | grep Mem)
        local used=$(echo "$memory_info" | awk '{print $3}')
        local total=$(echo "$memory_info" | awk '{print $2}')
        local available=$(echo "$memory_info" | awk '{print $7}')
        log_success "Paměť: $used/$total použito, $available dostupné"
    else
        log_warning "Nelze zjistit stav paměti"
    fi
}

show_logs() {
    log_info "Posledních 20 řádků logů..."
    
    if pgrep -f "streamlit.*8505" >/dev/null; then
        local pid=$(pgrep -f "streamlit.*8505")
        echo "--- Streamlit Process Info ---"
        ps -p "$pid" -o pid,ppid,cmd,etime,pcpu,pmem 2>/dev/null || echo "Nelze získat info o procesu"
        echo ""
    fi
    
    # Zkusit najít log soubory
    if [[ -f "/tmp/streamlit.log" ]]; then
        echo "--- Streamlit Logs ---"
        tail -20 "/tmp/streamlit.log"
    elif [[ -d "/root/.streamlit/logs" ]]; then
        echo "--- Streamlit Logs ---"
        find "/root/.streamlit/logs" -name "*.log" -exec tail -10 {} \;
    else
        echo "--- System Logs (dmesg) ---"
        dmesg | tail -10 2>/dev/null || echo "Nelze získat system logy"
    fi
}

run_diagnostics() {
    echo "==========================================="
    echo "    LoRA Tuymans Diagnostics Report"
    echo "==========================================="
    echo "Čas: $(date)"
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo "Working Directory: $(pwd)"
    echo "==========================================="
    echo ""
    
    check_streamlit
    echo ""
    check_port
    echo ""
    check_health
    echo ""
    check_gpu
    echo ""
    check_disk_space
    echo ""
    check_memory
    echo ""
    show_logs
    echo ""
    echo "==========================================="
    echo "           Diagnostics Complete"
    echo "==========================================="
}

# Continuous monitoring mode
monitor_continuous() {
    log_info "Spouštím kontinuální monitoring (Ctrl+C pro ukončení)..."
    
    while true; do
        clear
        run_diagnostics
        sleep 30
    done
}

# Main script logic
case "${1:-}" in
    "continuous"|"watch")
        monitor_continuous
        ;;
    "health")
        check_health
        ;;
    "gpu")
        check_gpu
        ;;
    "logs")
        show_logs
        ;;
    *)
        run_diagnostics
        ;;
esac
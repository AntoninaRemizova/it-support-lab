from fastmcp import FastMCP

mcp = FastMCP("IT-Support-Tools")

# Возвращает статус сервиса
@mcp.tool()
def get_server_status(service_name: str) -> str:
    services = {
        "vpn": "Работает",
        "email": "Работает",
        "3d-printer": "Не в сети"
    }
    return services.get(service_name.lower(), "Сервис не найден.")

if __name__ == "__main__":
    mcp.run(transport="stdio")
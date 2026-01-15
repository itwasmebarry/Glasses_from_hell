#include <WiFi.h>
#include <WebServer.h>

// WiFi Settings
const char* ssid     = "XIAO_YOLO_Band";
const char* password = "12345678";

// GPIO Pin Definitions
const int MOTOR_PIN = D10; // Connect all 5 motors in parallel here
const int LED_PIN   = D9;  // Connect all 5 LEDs in parallel here

WebServer server(80);

String getHTML() {
  String html = "<!DOCTYPE html><html>";
  html += "<head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">";
  html += "<style>body{font-family:Arial; text-align:center; background:#111; color:white; padding-top:50px;}";
  html += ".btn{display:inline-block; padding:25px 50px; font-size:28px; color:#fff; border-radius:15px; text-decoration:none; margin:15px;}";
  html += ".on{background-color:#2ecc71;} .off{background-color:#e74c3c;}</style></head><body>";
  html += "<h1>YOLO Band Master</h1>";
  html += "<a href=\"/on\" class=\"btn on\">SYSTEM ON</a><br>";
  html += "<a href=\"/off\" class=\"btn off\">SYSTEM OFF</a>";
  html += "</body></html>";
  return html;
}

void handleRoot() { server.send(200, "text/html", getHTML()); }

void handleOn() {
  digitalWrite(MOTOR_PIN, HIGH);
  digitalWrite(LED_PIN, HIGH);
  Serial.println("YOLO Mode: All Systems GO");
  server.send(200, "text/html", getHTML());
}

void handleOff() {
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  Serial.println("YOLO Mode: All Systems STOP");
  server.send(200, "text/html", getHTML());
}

void setup() {
  Serial.begin(115200);
  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  WiFi.softAP(ssid, password);
  Serial.println("Access Point Started");
  Serial.print("Connect to: "); Serial.println(ssid);
  Serial.print("IP: "); Serial.println(WiFi.softAPIP());

  server.on("/", handleRoot);
  server.on("/on", handleOn);
  server.on("/off", handleOff);
  server.begin();
}

void loop() {
  server.handleClient();
}
# Java (OkHttp) example

import java.io.IOException;
import java.util.*;
import okhttp3.*;

public class MapperClient {
  public static void main(String[] args) throws IOException {
    String baseUrl = System.getenv().getOrDefault("MAPPER_BASE_URL", "http://localhost:8000");
    String apiKey = System.getenv().getOrDefault("MAPPER_API_KEY", "YOUR_API_KEY");
    String tenantId = System.getenv().getOrDefault("MAPPER_TENANT_ID", "YOUR_TENANT_ID");

    OkHttpClient client = new OkHttpClient();

    String json = "{" +
      "\"detector\":\"orchestrated-multi\"," +
      "\"output\":\"toxic|hate|pii_detected\"," +
      "\"tenant_id\":\"" + tenantId + "\"," +
      "\"metadata\":{\"contributing_detectors\":[\"deberta-toxicity\",\"openai-moderation\"],\"aggregation_method\":\"weighted_average\",\"coverage_achieved\":1.0,\"provenance\":[{\"detector\":\"deberta-toxicity\",\"confidence\":0.93}]}}";

    RequestBody body = RequestBody.create(json, MediaType.parse("application/json"));
    Request request = new Request.Builder()
        .url(baseUrl + "/map")
        .addHeader("Content-Type", "application/json")
        .addHeader("X-API-Key", apiKey)
        .addHeader("X-Tenant-ID", tenantId)
        .addHeader("Idempotency-Key", "example-req-123")
        .post(body)
        .build();

    try (Response response = client.newCall(request).execute()) {
      String responseBody = response.body() != null ? response.body().string() : "";
      System.out.println(response.code());
      System.out.println(responseBody);
    }
  }
}

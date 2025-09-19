# C# (HttpClient) example

using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

class MapperClient
{
    static async Task Main()
    {
        var baseUrl = Environment.GetEnvironmentVariable("MAPPER_BASE_URL") ?? "http://localhost:8000";
        var apiKey = Environment.GetEnvironmentVariable("MAPPER_API_KEY") ?? "YOUR_API_KEY";
        var tenantId = Environment.GetEnvironmentVariable("MAPPER_TENANT_ID") ?? "YOUR_TENANT_ID";

        var json = "{\"detector\":\"orchestrated-multi\",\"output\":\"toxic|hate|pii_detected\",\"tenant_id\":\"" + tenantId + "\",\"metadata\":{\"contributing_detectors\":[\"deberta-toxicity\",\"openai-moderation\"],\"aggregation_method\":\"weighted_average\",\"coverage_achieved\":1.0,\"provenance\":[{\"detector\":\"deberta-toxicity\",\"confidence\":0.93}]}}";

        using var client = new HttpClient();
        var req = new HttpRequestMessage(HttpMethod.Post, baseUrl + "/map");
        req.Content = new StringContent(json, Encoding.UTF8, "application/json");
        req.Headers.Add("X-API-Key", apiKey);
        req.Headers.Add("X-Tenant-ID", tenantId);
        req.Headers.Add("Idempotency-Key", "example-req-123");

        var resp = await client.SendAsync(req);
        var body = await resp.Content.ReadAsStringAsync();
        Console.WriteLine(resp.StatusCode);
        Console.WriteLine(body);
    }
}

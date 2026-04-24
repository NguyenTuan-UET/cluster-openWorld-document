// API Service — gọi backend FastAPI
const API_BASE = 'http://localhost:8000';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API Error ${res.status}: ${err}`);
  }
  return res.json() as Promise<T>;
}

export async function getState() {
  return handleResponse(await fetch(`${API_BASE}/state`));
}

export async function resetState() {
  return handleResponse(await fetch(`${API_BASE}/reset`, { method: 'POST' }));
}

export async function extractAndCluster(texts: string[], fileNames?: string[]) {
  return handleResponse(
    await fetch(`${API_BASE}/process-and-cluster`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts, file_names: fileNames }),
    }),
  );
}

/** Gửi 1 tài liệu duy nhất — dùng để cập nhật tiến trình từng doc một */
export async function extractAndClusterOne(text: string, fileName: string) {
  return handleResponse<{ final_clusters: any[]; all_documents: any[] }>(
    await fetch(`${API_BASE}/process-and-cluster`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: [text], file_names: [fileName] }),
    }),
  );
}


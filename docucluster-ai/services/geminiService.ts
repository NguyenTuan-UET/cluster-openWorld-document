
import { GoogleGenAI, Type } from "@google/genai";
import { AnalyzedDocument, DocumentCluster } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

// Helper to convert File to base64 or text
export const fileToData = (file: File): Promise<{ base64: string; isPdf: boolean; text?: string }> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    const isPdf = file.type === 'application/pdf';
    
    if (isPdf) {
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        resolve({ base64: result.split(',')[1], isPdf: true });
      };
    } else {
      reader.readAsText(file);
      reader.onload = () => {
        resolve({ base64: '', isPdf: false, text: reader.result as string });
      };
    }
    reader.onerror = error => reject(error);
  });
};

/**
 * STEP 1: Extracts keyphrases and summary from a single document.
 */
const extractInfoFromDocument = async (
  file: File,
  id: string,
): Promise<AnalyzedDocument> => {
  const { base64, isPdf, text } = await fileToData(file);

  // FIX: Removed Schema type annotation as it's not a public export.
  const responseSchema = {
    type: Type.OBJECT,
    properties: {
      keyphrases: { type: Type.ARRAY, items: { type: Type.STRING } },
      summary: { type: Type.STRING }
    },
    required: ["keyphrases", "summary"]
  };

  const contentParts: any[] = [{ text: `Extract 5-10 keyphrases and a one-sentence summary from this document.` }];
  if (isPdf) {
    contentParts.unshift({ inlineData: { mimeType: 'application/pdf', data: base64 } });
  } else {
    contentParts.unshift({ text: `Document Content: ${text}` });
  }

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: { parts: contentParts },
    config: { responseMimeType: "application/json", responseSchema, temperature: 0.1 }
  });

  const json = JSON.parse(response.text || '{}');
  return { id, fileName: file.name, fileSize: file.size, type: file.type || 'text/plain', keyphrases: json.keyphrases || [], summary: json.summary || "" };
};


/**
 * STEP 2: Tries to assign new documents to existing clusters.
 */
const assignToExistingClusters = async (
  newDocuments: AnalyzedDocument[],
  existingClusters: DocumentCluster[]
): Promise<{ assignments: Record<string, string>, unassigned: AnalyzedDocument[] }> => {
  const prompt = `You are a document classification expert. Your task is to assign new documents to the most appropriate existing cluster.

EXISTING CLUSTERS:
${existingClusters.map(c => `- "${c.label}": Represents topics like [${c.documents.slice(0, 3).flatMap(d => d.keyphrases).join(', ')}]`).join('\n')}

NEW DOCUMENTS:
${newDocuments.map(d => `- ID: "${d.id}", Keyphrases: [${d.keyphrases.join(', ')}]`).join('\n')}

INSTRUCTIONS:
- For each new document, find the best matching cluster label from the existing list.
- If a document does not fit well into ANY existing cluster, you MUST assign its label as null. Be strict.
- Return a single JSON object with the assignments.

OUTPUT FORMAT:
{
  "assignments": [
    { "documentId": "...", "clusterLabel": "<label_name_or_null>" }
  ]
}
`;

  const responseSchema = {
    type: Type.OBJECT,
    properties: {
      assignments: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            documentId: { type: Type.STRING },
            clusterLabel: { type: Type.STRING, nullable: true }
          },
          required: ["documentId", "clusterLabel"]
        }
      }
    },
    required: ["assignments"]
  };

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: prompt,
    config: { responseMimeType: "application/json", responseSchema }
  });

  const result = JSON.parse(response.text || '{"assignments":[]}');
  const assignments: Record<string, string> = {};
  const unassignedIds = new Set(newDocuments.map(d => d.id));

  for (const item of result.assignments) {
    if (item.clusterLabel && existingClusters.some(c => c.label === item.clusterLabel)) {
      assignments[item.documentId] = item.clusterLabel;
      unassignedIds.delete(item.documentId);
    }
  }

  const unassigned = newDocuments.filter(d => unassignedIds.has(d.id));
  return { assignments, unassigned };
};

/**
 * STEP 3: Clusters a batch of unassigned documents to create NEW clusters.
 */
const clusterUnassignedDocuments = async (
  documents: AnalyzedDocument[]
): Promise<DocumentCluster[]> => {
  if (documents.length === 0) return [];
  if (documents.length === 1) return [{ label: documents[0].keyphrases[0] || "Tài liệu đơn lẻ", documents }];

  const clusteringPrompt = `You are an expert in document clustering. Group the following documents into meaningful clusters based on their keyphrases and assign a short, descriptive Vietnamese label to each cluster.

Rules:
- Cluster labels MUST be in Vietnamese (e.g., "Nghiên cứu AI", "Báo cáo tài chính").
- Use as few clusters as possible while keeping topics distinct.
- Do NOT explain your reasoning.

Input:
${JSON.stringify(documents.map(d => ({ id: d.id, keyphrases: d.keyphrases })), null, 2)}

Output Format:
{
  "clusters": [
    { "label": "<vietnamese_label>", "documents": ["id1", "id2"] }
  ]
}
`;

  // FIX: Removed Schema type annotation as it's not a public export.
  const responseSchema = {
    type: Type.OBJECT,
    properties: {
      clusters: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            label: { type: Type.STRING },
            documents: { type: Type.ARRAY, items: { type: Type.STRING } }
          },
          required: ["label", "documents"]
        }
      }
    },
    required: ["clusters"]
  };

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: clusteringPrompt,
    config: { responseMimeType: "application/json", responseSchema }
  });

  const result = JSON.parse(response.text || '{"clusters": []}');
  const docMap = new Map(documents.map(doc => [doc.id, doc]));
  
  const hydratedClusters: DocumentCluster[] = result.clusters.map((cluster: any) => ({
    label: cluster.label,
    documents: cluster.documents.map((docId: string) => docMap.get(docId)).filter(Boolean) as AnalyzedDocument[]
  }));

  const clusteredDocIds = new Set(hydratedClusters.flatMap(c => c.documents.map(d => d.id)));
  const missedDocs = documents.filter(d => !clusteredDocIds.has(d.id));
  if (missedDocs.length > 0) {
    hydratedClusters.push({ label: "Linh tinh", documents: missedDocs });
  }

  return hydratedClusters;
};

/**
 * MAIN WORKFLOW FUNCTION
 */
export const processAndClusterNewDocuments = async (
  files: File[],
  existingClusters: DocumentCluster[],
  existingDocuments: AnalyzedDocument[]
): Promise<{ finalClusters: DocumentCluster[], allDocuments: AnalyzedDocument[] }> => {
  // Step 1: Extract info from new files
  const newlyAnalyzedDocs = await Promise.all(
    files.map((file, index) => {
      const docId = `doc-${Date.now()}-${index}`;
      return extractInfoFromDocument(file, docId);
    })
  );

  let assignments: Record<string, string> = {};
  let unassignedDocs = newlyAnalyzedDocs;
  
  // Step 2: If clusters exist, try to assign new docs to them
  if (existingClusters.length > 0 && newlyAnalyzedDocs.length > 0) {
    const result = await assignToExistingClusters(newlyAnalyzedDocs, existingClusters);
    assignments = result.assignments;
    unassignedDocs = result.unassigned;
  }
  
  // Step 3: Cluster the remaining unassigned documents to form new clusters
  const newClusters = await clusterUnassignedDocuments(unassignedDocs);

  // Step 4: Merge results
  const finalClustersMap = new Map<string, DocumentCluster>();

  // Add existing clusters
  for (const cluster of existingClusters) {
    finalClustersMap.set(cluster.label, { ...cluster, documents: [...cluster.documents] });
  }

  // Add newly assigned documents to existing clusters
  const newDocsMap = new Map(newlyAnalyzedDocs.map(d => [d.id, d]));
  for (const docId in assignments) {
    const clusterLabel = assignments[docId];
    const doc = newDocsMap.get(docId);
    if (doc && finalClustersMap.has(clusterLabel)) {
      finalClustersMap.get(clusterLabel)!.documents.push(doc);
    }
  }

  // Add newly created clusters
  for (const newCluster of newClusters) {
    // Avoid label collision, though unlikely
    const newLabel = finalClustersMap.has(newCluster.label) ? `${newCluster.label} (Mới)` : newCluster.label;
    finalClustersMap.set(newLabel, newCluster);
  }

  const finalClusters = Array.from(finalClustersMap.values());
  const allDocuments = [...existingDocuments, ...newlyAnalyzedDocs];

  return { finalClusters, allDocuments };
};

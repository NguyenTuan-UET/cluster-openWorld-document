
export enum ProcessingStatus {
  IDLE = 'IDLE',
  EXTRACTING = 'EXTRACTING', // Changed from PROCESSING
  CLUSTERING = 'CLUSTERING', // Changed from REFINING
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR'
}

// Represents a document after initial keyphrase extraction
export interface AnalyzedDocument {
  id: string;
  fileName: string;
  fileSize: number;
  type: string;
  keyphrases: string[];
  summary: string;
}

// Represents the final output after the clustering step
export interface DocumentCluster {
  label: string;
  documents: AnalyzedDocument[];
}

export interface ProcessingError {
  fileName?: string;
  message: string;
}

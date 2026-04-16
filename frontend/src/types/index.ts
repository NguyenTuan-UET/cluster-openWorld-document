export enum ProcessingStatus {
  IDLE = 'IDLE',
  EXTRACTING = 'EXTRACTING',
  CLUSTERING = 'CLUSTERING',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR',
}

export interface AnalyzedDocument {
  id: string;
  fileName: string;
  keyphrases: string[];
  summary: string;
}

export interface DocumentCluster {
  label: string;
  documents: AnalyzedDocument[];
}

export interface ProcessResult {
  final_clusters: DocumentCluster[];
  all_documents: AnalyzedDocument[];
}

export interface State {
  documents: AnalyzedDocument[];
  clusters: DocumentCluster[];
  status: ProcessingStatus;
}

export interface ProcessingError {
  fileName?: string;
  message: string;
}

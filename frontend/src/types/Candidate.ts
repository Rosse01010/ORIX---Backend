/**
 * A person that the backend considers a possible match
 * for an unidentified or off-angle face.
 */
export interface CandidateMatch {
  person_id: string;
  name: string;
  similarity: number;   // cosine similarity 0–1
}

/**
 * One entry in the similarity panel.
 * Represents an unidentified face with its top candidate matches.
 */
export interface FaceCandidate {
  /** Index matching bboxes[face_index] in the detection event */
  face_index: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
    name: string;
    confidence: number;
    angle: string;
  };
  is_unknown: boolean;
  yaw: number;
  top_matches: CandidateMatch[];
  /** Which camera this face came from */
  cameraId: string;
  /** Original 512-dim embedding for confirmation */
  embedding?: number[];
  timestamp: string;
}

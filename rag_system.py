# rag_system.py
import os
import json
import glob
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmallMoleculeRAG")

# Load environment variables
load_dotenv()

class SmallMoleculeRAG:
    def __init__(self, docs_dir: str = "rag-system/documents"):
        """Initialize the RAG system with the path to documents."""
        self.client = OpenAI()
        self.docs_dir = docs_dir
        self.file_ids = {}
        self.processed_files = set()
        
        # Create directories if they don't exist
        os.makedirs(docs_dir, exist_ok=True)
        os.makedirs(f"{docs_dir}/papers", exist_ok=True)
        os.makedirs(f"{docs_dir}/notes", exist_ok=True)
        
        # Try to load existing file mapping
        self.mapping_file = "rag-system/file_mapping.json"
        self._load_file_mapping()
        
        logger.info(f"Initialized SmallMoleculeRAG with {len(self.file_ids)} processed files")
    
    def _load_file_mapping(self):
        """Load existing file ID mapping if available."""
        try:
            with open(self.mapping_file, 'r') as f:
                self.file_ids = json.load(f)
                for file_path in self.file_ids:
                    self.processed_files.add(file_path)
            logger.info(f"Loaded {len(self.file_ids)} file mappings")
        except FileNotFoundError:
            self.file_ids = {}
            logger.warning("No existing file mapping found")
    
    def _save_file_mapping(self):
        """Save file ID mapping to disk."""
        os.makedirs(os.path.dirname(self.mapping_file), exist_ok=True)
        with open(self.mapping_file, 'w') as f:
            json.dump(self.file_ids, f, indent=2)
        logger.info("Saved file mapping")
    
    def upload_documents(self, file_pattern: Optional[str] = None):
        """
        Upload documents to OpenAI and store their file IDs.
        
        Args:
            file_pattern: Optional glob pattern to match specific files
        """
        # Default pattern matches all PDFs
        if not file_pattern:
            file_pattern = f"{self.docs_dir}/**/*.pdf"
        
        files = glob.glob(file_pattern, recursive=True)
        logger.info(f"Found {len(files)} files matching pattern")
        
        new_files = 0
        for file_path in files:
            # Skip already processed files
            if file_path in self.processed_files:
                logger.info(f"Skipping already processed file: {file_path}")
                continue
            
            try:
                logger.info(f"Uploading: {file_path}")
                with open(file_path, "rb") as f:
                    file = self.client.files.create(
                        file=f,
                        purpose="user_data"
                    )
                
                # Store the file ID
                self.file_ids[file_path] = file.id
                self.processed_files.add(file_path)
                new_files += 1
                
                # Save mapping after each successful upload
                self._save_file_mapping()
                
                # Rate limiting - be nice to the API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Uploaded {new_files} new files")
    
    def get_answer(self, question: str, file_path: Optional[str] = None, 
                  model: str = "gpt-4o") -> Dict[str, Any]:
        """
        Query the system with a question.
        
        Args:
            question: The question to ask
            file_path: Optional specific file to query (uses all files if None)
            model: Model to use (default: gpt-4o)
            
        Returns:
            Dictionary with the response
        """
        if not self.file_ids:
            logger.warning("No documents have been uploaded yet")
            return {
                "question": question,
                "answer": "No documents have been uploaded yet. Please upload documents first.",
                "error": "no_documents"
            }
            
        if file_path and file_path in self.file_ids:
            # Query a specific file
            file_id = self.file_ids[file_path]
            logger.info(f"Querying file: {os.path.basename(file_path)}")
            
            try:
                response = self.client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "file_id": file_id,
                                },
                                {
                                    "type": "input_text",
                                    "text": f"""
                                    Question: {question}
                                    
                                    Please provide a detailed answer with citations to specific parts of the document.
                                    If the information is not in the document, please state that clearly.
                                    """,
                                }
                            ]
                        }
                    ]
                )
                
                return {
                    "question": question,
                    "answer": response.output_text,
                    "source": os.path.basename(file_path),
                    "model": model
                }
            except Exception as e:
                logger.error(f"Error querying file {file_path}: {str(e)}")
                return {
                    "question": question,
                    "answer": f"Error querying the document: {str(e)}",
                    "error": "query_error"
                }
            
        elif not file_path and self.file_ids:
            # Query across all files
            logger.info("Querying across all files")
            
            try:
                # We need to implement a workaround since we can't query multiple files at once
                # with the current API - we'll query each file and combine results
                
                # Start with a system that keeps track of the best answer
                best_answer = None
                best_confidence = 0
                sources = []
                
                for path, file_id in self.file_ids.items():
                    try:
                        logger.info(f"Querying file: {os.path.basename(path)}")
                        
                        response = self.client.responses.create(
                            model=model,
                            input=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_file",
                                            "file_id": file_id,
                                        },
                                        {
                                            "type": "input_text",
                                            "text": f"""
                                            Question: {question}
                                            
                                            First, rate your confidence in answering this question on a scale of 0-10 based on this document.
                                            Then, if your confidence is above 3, provide a detailed answer with citations.
                                            Format your response exactly like this:
                                            
                                            Confidence: [0-10]
                                            
                                            Answer: [Your detailed answer with citations]
                                            """,
                                        }
                                    ]
                                }
                            ]
                        )
                        
                        # Parse the response
                        response_text = response.output_text
                        
                        # Extract confidence rating
                        if "Confidence:" in response_text:
                            confidence_line = response_text.split("Confidence:")[1].split("\n")[0].strip()
                            try:
                                confidence = float(confidence_line)
                            except ValueError:
                                # If we can't parse the confidence, assume it's 0
                                confidence = 0
                        else:
                            confidence = 0
                        
                        # If this is a confident answer, consider it for the best answer
                        if confidence > 3 and confidence > best_confidence:
                            best_confidence = confidence
                            # Extract just the answer part
                            if "Answer:" in response_text:
                                best_answer = response_text.split("Answer:")[1].strip()
                            else:
                                best_answer = response_text
                            
                            # Track the source
                            sources.append(os.path.basename(path))
                        
                    except Exception as e:
                        logger.error(f"Error querying {path}: {str(e)}")
                
                # If we found at least one good answer
                if best_answer:
                    return {
                        "question": question,
                        "answer": best_answer,
                        "sources": sources,
                        "confidence": best_confidence,
                        "model": model
                    }
                else:
                    # If no good answers found, let's ask one more general question
                    return self._get_general_answer(question, model)
            except Exception as e:
                logger.error(f"Error querying across files: {str(e)}")
                return {
                    "question": question,
                    "answer": f"Error querying the documents: {str(e)}",
                    "error": "query_error"
                }
        else:
            logger.warning("No documents have been uploaded yet")
            return {
                "question": question,
                "answer": "No documents have been uploaded yet. Please upload documents first.",
                "error": "no_documents"
            }
    
    def _get_general_answer(self, question: str, model: str = "gpt-4o") -> Dict[str, Any]:
        """Get a general answer when no specific document has the answer."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a helpful assistant specializing in small molecule discovery
                                and computational chemistry with expertise in neuroscience applications.
                                If you don't know the answer, please say so clearly."""
                },
                {
                    "role": "user",
                    "content": f"""
                    Question: {question}
                    
                    Note: This question couldn't be answered confidently from the documents in the system.
                    Please provide your best general knowledge answer, but clearly indicate that this is not
                    from the document collection.
                    """
                }
            ]
        )
        
        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "sources": ["general_knowledge"],
            "model": model
        }
    
    def get_molecular_property_prediction(self, smiles: str, properties: List[str] = None) -> Dict[str, Any]:
        """
        Get predictions for molecular properties based on the documents.
        
        Args:
            smiles: SMILES string of the molecule
            properties: List of properties to predict (default: common drug-like properties)
            
        Returns:
            Dictionary with property predictions and explanations
        """
        if properties is None:
            properties = [
                "logP", "molecular_weight", "hydrogen_bond_donors", 
                "hydrogen_bond_acceptors", "rotatable_bonds", 
                "blood_brain_barrier_permeability", "toxicity"
            ]
        
        question = f"""
        For the molecule with SMILES: {smiles}
        
        Please predict the following properties and explain your reasoning:
        {', '.join(properties)}
        
        Also, discuss any CNS-specific considerations for this molecule.
        """
        
        return self.get_answer(question)
    
    def get_molecular_similarity(self, smiles1: str, smiles2: str) -> Dict[str, Any]:
        """
        Analyze the similarity between two molecules based on the documents.
        
        Args:
            smiles1: SMILES string of the first molecule
            smiles2: SMILES string of the second molecule
            
        Returns:
            Dictionary with similarity analysis
        """
        question = f"""
        Compare the following two molecules:
        
        Molecule 1: {smiles1}
        Molecule 2: {smiles2}
        
        Please analyze their structural similarity, potential pharmacological properties,
        and CNS-specific considerations. How might their biological activities differ?
        """
        
        return self.get_answer(question)
    
    def get_target_prediction(self, smiles: str) -> Dict[str, Any]:
        """
        Predict potential biological targets for a molecule based on the documents.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            Dictionary with target predictions and explanations
        """
        question = f"""
        For the molecule with SMILES: {smiles}
        
        Please predict potential biological targets, focusing on CNS targets.
        Discuss the evidence from the literature for these predictions.
        What are the potential therapeutic applications?
        """
        
        return self.get_answer(question)
    
    def add_note(self, content: str, filename: str = None):
        """
        Add a personal note to the system.
        
        Args:
            content: Note content
            filename: Optional filename (will generate one if not provided)
        """
        if filename is None:
            filename = f"note_{int(time.time())}.txt"
        
        if not filename.endswith(".txt"):
            filename += ".txt"
        
        filepath = os.path.join(self.docs_dir, "notes", filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Note saved to {filepath}")
        
        # Upload the note to OpenAI
        self.upload_documents(file_pattern=filepath)
        
        return filepath


def create_cli():
    """Create a simple command-line interface for the RAG system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Small Molecule Discovery RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload documents")
    upload_parser.add_argument("--pattern", type=str, help="File pattern to match")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--question", "-q", type=str, help="Question to ask")
    query_parser.add_argument("--file", "-f", type=str, help="Specific file to query")
    query_parser.add_argument("--model", "-m", type=str, default="gpt-4o", 
                             help="Model to use (default: gpt-4o)")
    
    # Molecular property prediction command
    property_parser = subparsers.add_parser("property", help="Predict molecular properties")
    property_parser.add_argument("--smiles", "-s", type=str, required=True, help="SMILES string")
    property_parser.add_argument("--properties", "-p", type=str, nargs="+", 
                                help="Properties to predict (default: common drug-like properties)")
    
    # Molecular similarity command
    similarity_parser = subparsers.add_parser("similarity", help="Compare two molecules")
    similarity_parser.add_argument("--smiles1", "-s1", type=str, required=True, help="SMILES string of first molecule")
    similarity_parser.add_argument("--smiles2", "-s2", type=str, required=True, help="SMILES string of second molecule")
    
    # Target prediction command
    target_parser = subparsers.add_parser("target", help="Predict biological targets")
    target_parser.add_argument("--smiles", "-s", type=str, required=True, help="SMILES string")
    
    # Add note command
    note_parser = subparsers.add_parser("note", help="Add a note")
    note_parser.add_argument("--content", "-c", type=str, help="Note content")
    note_parser.add_argument("--file", "-f", type=str, help="Output filename")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag = SmallMoleculeRAG()
    
    if args.command == "upload":
        rag.upload_documents(file_pattern=args.pattern)
    
    elif args.command == "query":
        if args.question:
            result = rag.get_answer(args.question, file_path=args.file, model=args.model)
            print("\n" + "="*50)
            print(f"Question: {result['question']}")
            print("="*50)
            print(f"Answer: {result['answer']}")
            print("="*50)
            if "sources" in result:
                print(f"Sources: {', '.join(result['sources'])}")
            elif "source" in result:
                print(f"Source: {result['source']}")
        else:
            print("Please provide a question with --question")
    
    elif args.command == "property":
        if args.smiles:
            result = rag.get_molecular_property_prediction(args.smiles, args.properties)
            print("\n" + "="*50)
            print(f"Molecule: {args.smiles}")
            print("="*50)
            print(f"Predictions: {result['answer']}")
            print("="*50)
            if "sources" in result:
                print(f"Sources: {', '.join(result['sources'])}")
            elif "source" in result:
                print(f"Source: {result['source']}")
    
    elif args.command == "similarity":
        if args.smiles1 and args.smiles2:
            result = rag.get_molecular_similarity(args.smiles1, args.smiles2)
            print("\n" + "="*50)
            print(f"Molecule 1: {args.smiles1}")
            print(f"Molecule 2: {args.smiles2}")
            print("="*50)
            print(f"Analysis: {result['answer']}")
            print("="*50)
            if "sources" in result:
                print(f"Sources: {', '.join(result['sources'])}")
            elif "source" in result:
                print(f"Source: {result['source']}")
    
    elif args.command == "target":
        if args.smiles:
            result = rag.get_target_prediction(args.smiles)
            print("\n" + "="*50)
            print(f"Molecule: {args.smiles}")
            print("="*50)
            print(f"Target Predictions: {result['answer']}")
            print("="*50)
            if "sources" in result:
                print(f"Sources: {', '.join(result['sources'])}")
            elif "source" in result:
                print(f"Source: {result['source']}")
    
    elif args.command == "note":
        if args.content:
            filepath = rag.add_note(args.content, filename=args.file)
            print(f"Note added and uploaded: {filepath}")
        else:
            print("Please provide note content with --content")
    
    elif args.command == "interactive":
        print("\nSmall Molecule Discovery RAG System - Interactive Mode")
        print("Type 'exit' to quit, 'upload' to upload documents, 'note' to add a note")
        print("="*50)
        
        while True:
            command = input("\nEnter command (query/property/similarity/target/upload/note/exit): ").strip().lower()
            
            if command == "exit":
                break
            
            elif command == "upload":
                pattern = input("Enter file pattern (leave empty for all PDFs): ").strip()
                rag.upload_documents(file_pattern=pattern if pattern else None)
            
            elif command == "note":
                content = input("Enter note content: ").strip()
                filename = input("Enter filename (optional): ").strip()
                if content:
                    rag.add_note(content, filename=filename if filename else None)
                else:
                    print("Note content cannot be empty")
            
            elif command == "query":
                question = input("Enter your question: ").strip()
                file_path = input("Specific file to query (optional): ").strip()
                model = input("Model to use (default: gpt-4o): ").strip() or "gpt-4o"
                
                if question:
                    result = rag.get_answer(
                        question, 
                        file_path=file_path if file_path else None,
                        model=model
                    )
                    print("\n" + "="*50)
                    print(f"Question: {result['question']}")
                    print("="*50)
                    print(f"Answer: {result['answer']}")
                    print("="*50)
                    if "sources" in result:
                        print(f"Sources: {', '.join(result['sources'])}")
                    elif "source" in result:
                        print(f"Source: {result['source']}")
                else:
                    print("Question cannot be empty")
            
            elif command == "property":
                smiles = input("Enter SMILES string: ").strip()
                properties = input("Enter properties to predict (comma-separated, leave empty for defaults): ").strip()
                
                if smiles:
                    props = properties.split(",") if properties else None
                    result = rag.get_molecular_property_prediction(smiles, props)
                    print("\n" + "="*50)
                    print(f"Molecule: {smiles}")
                    print("="*50)
                    print(f"Predictions: {result['answer']}")
                    print("="*50)
                    if "sources" in result:
                        print(f"Sources: {', '.join(result['sources'])}")
                    elif "source" in result:
                        print(f"Source: {result['source']}")
                else:
                    print("SMILES string cannot be empty")
            
            elif command == "similarity":
                smiles1 = input("Enter SMILES string of first molecule: ").strip()
                smiles2 = input("Enter SMILES string of second molecule: ").strip()
                
                if smiles1 and smiles2:
                    result = rag.get_molecular_similarity(smiles1, smiles2)
                    print("\n" + "="*50)
                    print(f"Molecule 1: {smiles1}")
                    print(f"Molecule 2: {smiles2}")
                    print("="*50)
                    print(f"Analysis: {result['answer']}")
                    print("="*50)
                    if "sources" in result:
                        print(f"Sources: {', '.join(result['sources'])}")
                    elif "source" in result:
                        print(f"Source: {result['source']}")
                else:
                    print("Both SMILES strings are required")
            
            elif command == "target":
                smiles = input("Enter SMILES string: ").strip()
                
                if smiles:
                    result = rag.get_target_prediction(smiles)
                    print("\n" + "="*50)
                    print(f"Molecule: {smiles}")
                    print("="*50)
                    print(f"Target Predictions: {result['answer']}")
                    print("="*50)
                    if "sources" in result:
                        print(f"Sources: {', '.join(result['sources'])}")
                    elif "source" in result:
                        print(f"Source: {result['source']}")
                else:
                    print("SMILES string cannot be empty")
            
            else:
                print("Unknown command. Available commands: query, property, similarity, target, upload, note, exit")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    create_cli()
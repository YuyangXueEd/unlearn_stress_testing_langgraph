# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from agent.rag_manager import initialize_rag_database_async, get_rag_database

# Set up logger
logger = logging.getLogger(__name__)


def _confirm_rag_db_sync():
    """Synchronous function to confirm RAG database connection and contents."""
    print("🔍 Confirming RAG database connection...")
    rag_db = get_rag_database()
    if rag_db:
        try:
            collection = rag_db.client.get_collection(rag_db.collection_name)
            count = collection.count()
            print(f"✅ RAG database connected successfully. Collection '{rag_db.collection_name}' contains {count} documents.")
            logger.info(f"RAG database connected. Collection '{rag_db.collection_name}' contains {count} documents.")
        except Exception as e:
            print(f"⚠️  Could not confirm RAG database contents: {e}")
            logger.error(f"Could not confirm RAG database contents: {e}")
    else:
        print("❌ RAG database instance not found after initialization.")
        logger.error("RAG database instance not found after initialization.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    print("🚀 Starting FastAPI application with RAG initialization...")
    await initialize_rag_database_async(
        persist_directory="chroma_db",
        papers_directory="paper"    # Corrected path for app startup
    )
    print("✅ FastAPI application startup complete")
    
    # Run synchronous confirmation in a separate thread
    await asyncio.to_thread(_confirm_rag_db_sync)
    
    yield
    
    # Shutdown (if needed)
    print("🛑 FastAPI application shutting down...")


# Define the FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)


def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # Return a dummy router if build isn't ready
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
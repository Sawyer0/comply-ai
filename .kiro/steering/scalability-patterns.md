---
inclusion: always
---

# Scalability Patterns & Architecture

## Event Sourcing for Audit Immutability

### Event Store Implementation
```python
# src/llama_mapper/events/event_store.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import uuid

@dataclass
class Event:
    """Base event class"""
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    event_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    version: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

class EventStore:
    """Event store for immutable audit trail"""
    
    def __init__(self, db_pool):
        self.db = db_pool
    
    async def append_event(self, event: Event) -> None:
        """Append event to store"""
        query = """
        INSERT INTO events (
            event_id, event_type, aggregate_id, aggregate_type,
            event_data, metadata, timestamp, version
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        async with self.db.acquire() as conn:
            await conn.execute(
                query,
                event.event_id,
                event.event_type,
                event.aggregate_id,
                event.aggregate_type,
                json.dumps(event.event_data),
                json.dumps(event.metadata),
                event.timestamp,
                event.version
            )
    
    async def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Get events for aggregate"""
        query = """
        SELECT * FROM events 
        WHERE aggregate_id = $1 AND version > $2
        ORDER BY version ASC
        """
        
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, aggregate_id, from_version)
        
        return [
            Event(
                event_id=row['event_id'],
                event_type=row['event_type'],
                aggregate_id=row['aggregate_id'],
                aggregate_type=row['aggregate_type'],
                event_data=json.loads(row['event_data']),
                metadata=json.loads(row['metadata']),
                timestamp=row['timestamp'],
                version=row['version']
            )
            for row in rows
        ]
    
    async def get_events_by_type(self, event_type: str, limit: int = 1000) -> List[Event]:
        """Get events by type for projections"""
        query = """
        SELECT * FROM events 
        WHERE event_type = $1
        ORDER BY timestamp DESC
        LIMIT $2
        """
        
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, event_type, limit)
        
        return [self._row_to_event(row) for row in rows]

# Domain Events
@dataclass
class DetectorExecutedEvent(Event):
    """Event for detector execution"""
    
    @classmethod
    def create(cls, detector_id: str, input_data: Dict[str, Any], 
               output_data: Dict[str, Any], execution_time: float) -> 'DetectorExecutedEvent':
        return cls(
            event_id=str(uuid.uuid4()),
            event_type="detector_executed",
            aggregate_id=detector_id,
            aggregate_type="detector",
            event_data={
                "input_hash": hash(str(input_data)),  # Don't store raw input
                "output_data": output_data,
                "execution_time": execution_time
            },
            metadata={
                "correlation_id": get_correlation_id(),
                "user_id": get_current_user_id(),
                "tenant_id": get_current_tenant_id()
            },
            timestamp=datetime.utcnow(),
            version=1
        )

@dataclass
class ComplianceMappingCreatedEvent(Event):
    """Event for compliance mapping creation"""
    
    @classmethod
    def create(cls, mapping_id: str, canonical_taxonomy: Dict[str, Any],
               framework_mappings: List[Dict[str, Any]]) -> 'ComplianceMappingCreatedEvent':
        return cls(
            event_id=str(uuid.uuid4()),
            event_type="compliance_mapping_created",
            aggregate_id=mapping_id,
            aggregate_type="compliance_mapping",
            event_data={
                "canonical_taxonomy": canonical_taxonomy,
                "framework_mappings": framework_mappings
            },
            metadata={
                "correlation_id": get_correlation_id(),
                "created_by": get_current_user_id(),
                "tenant_id": get_current_tenant_id()
            },
            timestamp=datetime.utcnow(),
            version=1
        )
```

### Event Projections
```python
# src/llama_mapper/events/projections.py
class EventProjection:
    """Base class for event projections"""
    
    def __init__(self, event_store: EventStore, db_pool):
        self.event_store = event_store
        self.db = db_pool
        self.last_processed_version = 0
    
    async def rebuild_projection(self):
        """Rebuild projection from events"""
        await self.clear_projection()
        
        events = await self.event_store.get_events_by_type(self.event_type)
        for event in events:
            await self.handle_event(event)
    
    async def handle_event(self, event: Event):
        """Handle individual event - to be implemented by subclasses"""
        raise NotImplementedError

class ComplianceAuditProjection(EventProjection):
    """Projection for compliance audit trail"""
    
    event_type = "compliance_mapping_created"
    
    async def handle_event(self, event: Event):
        """Handle compliance mapping events"""
        if event.event_type == "compliance_mapping_created":
            await self.create_audit_record(event)
    
    async def create_audit_record(self, event: Event):
        """Create audit record from event"""
        query = """
        INSERT INTO compliance_audit_trail (
            event_id, mapping_id, canonical_taxonomy, framework_mappings,
            created_by, tenant_id, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        async with self.db.acquire() as conn:
            await conn.execute(
                query,
                event.event_id,
                event.aggregate_id,
                json.dumps(event.event_data['canonical_taxonomy']),
                json.dumps(event.event_data['framework_mappings']),
                event.metadata['created_by'],
                event.metadata['tenant_id'],
                event.timestamp
            )
```

## CQRS (Command Query Responsibility Segregation)

### Command Side
```python
# src/llama_mapper/cqrs/commands.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class Command(ABC):
    """Base command class"""
    pass

class CreateComplianceMappingCommand(Command):
    """Command to create compliance mapping"""
    
    def __init__(self, detector_output: Dict[str, Any], 
                 framework: str, tenant_id: str):
        self.detector_output = detector_output
        self.framework = framework
        self.tenant_id = tenant_id

class CommandHandler(ABC):
    """Base command handler"""
    
    @abstractmethod
    async def handle(self, command: Command) -> Any:
        pass

class CreateComplianceMappingHandler(CommandHandler):
    """Handler for creating compliance mappings"""
    
    def __init__(self, mapper_service, event_store: EventStore):
        self.mapper_service = mapper_service
        self.event_store = event_store
    
    async def handle(self, command: CreateComplianceMappingCommand) -> str:
        """Handle compliance mapping creation"""
        # Execute business logic
        canonical_result = await self.mapper_service.map_to_canonical(
            command.detector_output
        )
        
        framework_mappings = await self.mapper_service.map_to_framework(
            canonical_result, command.framework
        )
        
        mapping_id = str(uuid.uuid4())
        
        # Create and store event
        event = ComplianceMappingCreatedEvent.create(
            mapping_id=mapping_id,
            canonical_taxonomy=canonical_result.to_dict(),
            framework_mappings=[m.to_dict() for m in framework_mappings]
        )
        
        await self.event_store.append_event(event)
        
        return mapping_id

class CommandBus:
    """Command bus for routing commands to handlers"""
    
    def __init__(self):
        self.handlers = {}
    
    def register_handler(self, command_type: type, handler: CommandHandler):
        """Register command handler"""
        self.handlers[command_type] = handler
    
    async def execute(self, command: Command) -> Any:
        """Execute command"""
        handler = self.handlers.get(type(command))
        if not handler:
            raise ValueError(f"No handler registered for {type(command)}")
        
        return await handler.handle(command)
```

### Query Side
```python
# src/llama_mapper/cqrs/queries.py
class Query(ABC):
    """Base query class"""
    pass

class GetComplianceHistoryQuery(Query):
    """Query for compliance history"""
    
    def __init__(self, tenant_id: str, framework: str, 
                 start_date: datetime, end_date: datetime):
        self.tenant_id = tenant_id
        self.framework = framework
        self.start_date = start_date
        self.end_date = end_date

class QueryHandler(ABC):
    """Base query handler"""
    
    @abstractmethod
    async def handle(self, query: Query) -> Any:
        pass

class GetComplianceHistoryHandler(QueryHandler):
    """Handler for compliance history queries"""
    
    def __init__(self, read_db_pool):
        self.read_db = read_db_pool  # Separate read database
    
    async def handle(self, query: GetComplianceHistoryQuery) -> List[Dict[str, Any]]:
        """Handle compliance history query"""
        sql = """
        SELECT 
            mapping_id,
            canonical_taxonomy,
            framework_mappings,
            created_by,
            timestamp
        FROM compliance_audit_trail
        WHERE tenant_id = $1 
          AND framework = $2
          AND timestamp BETWEEN $3 AND $4
        ORDER BY timestamp DESC
        """
        
        async with self.read_db.acquire() as conn:
            rows = await conn.fetch(
                sql, 
                query.tenant_id, 
                query.framework,
                query.start_date, 
                query.end_date
            )
        
        return [dict(row) for row in rows]

class QueryBus:
    """Query bus for routing queries to handlers"""
    
    def __init__(self):
        self.handlers = {}
    
    def register_handler(self, query_type: type, handler: QueryHandler):
        """Register query handler"""
        self.handlers[query_type] = handler
    
    async def execute(self, query: Query) -> Any:
        """Execute query"""
        handler = self.handlers.get(type(query))
        if not handler:
            raise ValueError(f"No handler registered for {type(query)}")
        
        return await handler.handle(query)
```

## Advanced Batch Processing

### Stream Processing with Apache Kafka
```python
# src/llama_mapper/streaming/kafka_processor.py
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import asyncio
import json
from typing import List, Dict, Any

class KafkaStreamProcessor:
    """Stream processor for real-time compliance analysis"""
    
    def __init__(self, bootstrap_servers: str):
        self.bootstrap_servers = bootstrap_servers
        self.consumer = None
        self.producer = None
    
    async def start(self):
        """Start Kafka consumer and producer"""
        self.consumer = AIOKafkaConsumer(
            'detector-outputs',
            bootstrap_servers=self.bootstrap_servers,
            group_id='compliance-mapper',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        await self.consumer.start()
        await self.producer.start()
    
    async def process_stream(self):
        """Process incoming detector outputs"""
        try:
            async for message in self.consumer:
                detector_output = message.value
                
                # Process in parallel batches
                await self.process_detector_output(detector_output)
                
        except Exception as e:
            logger.error("Stream processing error", error=str(e))
        finally:
            await self.consumer.stop()
            await self.producer.stop()
    
    async def process_detector_output(self, detector_output: Dict[str, Any]):
        """Process individual detector output"""
        try:
            # Map to canonical taxonomy
            canonical_result = await self.map_to_canonical(detector_output)
            
            # Generate compliance mappings
            compliance_mappings = await self.generate_compliance_mappings(canonical_result)
            
            # Publish results
            await self.producer.send(
                'compliance-mappings',
                {
                    'canonical_result': canonical_result,
                    'compliance_mappings': compliance_mappings,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error("Failed to process detector output", 
                        error=str(e), 
                        detector_output_id=detector_output.get('id'))
```

### Batch Processing Optimization
```python
# src/llama_mapper/batch/batch_processor.py
import asyncio
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class BatchProcessor:
    """Optimized batch processing for large datasets"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 10):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, items: List[Any], 
                           processor_func: Callable) -> List[Any]:
        """Process items in optimized batches"""
        # Split into batches
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches in parallel
        tasks = [
            self.process_single_batch(batch, processor_func)
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error("Batch processing failed", error=str(batch_result))
                continue
            results.extend(batch_result)
        
        return results
    
    async def process_single_batch(self, batch: List[Any], 
                                  processor_func: Callable) -> List[Any]:
        """Process a single batch"""
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive work in thread pool
        return await loop.run_in_executor(
            self.executor,
            self._process_batch_sync,
            batch,
            processor_func
        )
    
    def _process_batch_sync(self, batch: List[Any], 
                           processor_func: Callable) -> List[Any]:
        """Synchronous batch processing"""
        results = []
        for item in batch:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                logger.error("Item processing failed", 
                           error=str(e), 
                           item_id=getattr(item, 'id', 'unknown'))
        return results

# Specialized batch processors
class ComplianceBatchProcessor(BatchProcessor):
    """Specialized processor for compliance mappings"""
    
    def __init__(self, mapper_service, **kwargs):
        super().__init__(**kwargs)
        self.mapper_service = mapper_service
    
    async def process_compliance_batch(self, detector_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of detector outputs for compliance"""
        
        def process_single_output(output: Dict[str, Any]) -> Dict[str, Any]:
            # This runs in thread pool
            canonical_result = self.mapper_service.map_to_canonical_sync(output)
            framework_mappings = self.mapper_service.map_to_frameworks_sync(canonical_result)
            
            return {
                'input_id': output.get('id'),
                'canonical_result': canonical_result,
                'framework_mappings': framework_mappings,
                'processed_at': datetime.utcnow().isoformat()
            }
        
        return await self.process_batch(detector_outputs, process_single_output)
```

## Database Optimization Patterns

### Read Replicas and Connection Pooling
```python
# src/llama_mapper/database/connection_manager.py
import asyncpg
from typing import Dict, List
import asyncio

class DatabaseConnectionManager:
    """Advanced database connection management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.write_pool = None
        self.read_pools = {}
        self.connection_stats = {}
    
    async def initialize(self):
        """Initialize connection pools"""
        # Write pool (primary database)
        self.write_pool = await asyncpg.create_pool(
            **self.config['write_db'],
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Read pools (read replicas)
        for replica_name, replica_config in self.config['read_replicas'].items():
            self.read_pools[replica_name] = await asyncpg.create_pool(
                **replica_config,
                min_size=10,
                max_size=50,
                command_timeout=30
            )
    
    async def get_write_connection(self):
        """Get connection for write operations"""
        return self.write_pool.acquire()
    
    async def get_read_connection(self, preferred_replica: str = None):
        """Get connection for read operations with load balancing"""
        if preferred_replica and preferred_replica in self.read_pools:
            return self.read_pools[preferred_replica].acquire()
        
        # Load balance across read replicas
        replica_name = self._select_best_replica()
        return self.read_pools[replica_name].acquire()
    
    def _select_best_replica(self) -> str:
        """Select best read replica based on load"""
        # Simple round-robin for now, could be enhanced with health checks
        replica_names = list(self.read_pools.keys())
        if not hasattr(self, '_replica_index'):
            self._replica_index = 0
        
        replica_name = replica_names[self._replica_index]
        self._replica_index = (self._replica_index + 1) % len(replica_names)
        
        return replica_name
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all database connections"""
        health_status = {}
        
        # Check write pool
        try:
            async with self.write_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status['write_db'] = True
        except Exception:
            health_status['write_db'] = False
        
        # Check read replicas
        for replica_name, pool in self.read_pools.items():
            try:
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                health_status[f'read_replica_{replica_name}'] = True
            except Exception:
                health_status[f'read_replica_{replica_name}'] = False
        
        return health_status
```

These scalability patterns provide:

1. **Event Sourcing**: Immutable audit trail with event projections
2. **CQRS**: Separate read/write models for optimal performance
3. **Stream Processing**: Real-time processing with Kafka
4. **Batch Optimization**: Parallel processing for large datasets
5. **Database Scaling**: Read replicas with intelligent load balancing

This architecture can handle massive scale while maintaining data consistency and audit requirements!
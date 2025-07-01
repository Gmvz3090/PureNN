HEADER = purenn

.PHONY: install uninstall

install:
	@if [ "$$EUID" = "0" ] || [ "$$USER" = "root" ]; then \
		cp $(HEADER) /usr/local/include/; \
		echo "Installed to /usr/local/include/$(HEADER)"; \
	else \
		mkdir -p "$$HOME/.local/include"; \
		cp $(HEADER) "$$HOME/.local/include/"; \
		echo "Installed to $$HOME/.local/include/$(HEADER)"; \
	fi

uninstall:
	@if [ "$$EUID" = "0" ] || [ "$$USER" = "root" ]; then \
		rm -f /usr/local/include/$(HEADER); \
		echo "Removed from /usr/local/include/"; \
	else \
		rm -f "$$HOME/.local/include/$(HEADER)"; \
		echo "Removed from $$HOME/.local/include/"; \
	fi
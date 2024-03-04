SEMINAR := Seminar

.PHONY: seminar

seminar:
	@cp -r $(SEMINAR)/template $(SEMINAR)/$(shell date -u +%Y-%m-%d)

